#
# Copyright (C) 2023, Inria
# PEARL Lab at TU-Darmstadt. https://pearl-lab.com/
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact yufeng.jin@tu-darmstadt.de 
#

import os
import sys
import torch
import io
import copy
import numpy as np
import uuid
import wandb
import time
import cv2
import open3d as o3d
import hashlib
import random
import yaml
import datetime
import matplotlib.pyplot as plt
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss
from gaussian_renderer import render, renderer, network_gui
from scene import Scene, GaussianModel
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera
from utils.general_utils import safe_state, convert_depth_to_rgb, batch_images_from_numpy_to_tensor, image_from_numpy_to_tensor, get_img_from_fig
from tqdm import tqdm
from utils.camera_utils import calculate_fov_from_K
from utils.image_utils import psnr
from utils.graphics_utils import BasicPointCloud, getWorld2View2
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, GroupParams
from open3d.geometry import PointCloud
from pdb import set_trace
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix

# TODO learning gaussian in unit coordinate or world coordiate, test on blender data
class ConfigParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        
class GaussianSplatRunner:
    @classmethod
    def getNerfppNorm(cls, w2cList):
        """
        This is a function to approximate the center offset and radius of cameras.

        Parameters:
        - cam_info: a list of Camera.
        """
        def get_center_and_diag(cam_centers):
            cam_centers = np.hstack(cam_centers)
            avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
            center = avg_cam_center
            dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
            diagonal = np.max(dist)
            return center.flatten(), diagonal

        cam_centers = []

        for W2C in w2cList:
            C2W = np.linalg.inv(W2C)
            cam_centers.append(C2W[:3, 3:4])

        center, diagonal = get_center_and_diag(cam_centers)
        radius = diagonal * 1.1

        translate = -center

        return {"translate": translate, "radius": radius}
    
    @staticmethod
    def c2w_to_w2c(cam_quat, cam_t):
        assert cam_quat.shape[0] == cam_t.shape[0], 'Error: batch size of camera quaternion does not match the batch size of camera trans'
        batch_size = cam_quat.shape[0]

        c2w = torch.eye(4).cuda()
        c2w = c2w.unsqueeze(0).repeat(batch_size, 1, 1)


        c2w[:, :3, :3] = quaternion_to_matrix(cam_quat)
        c2w[:, :3, 3] = cam_t
        c2w[:, :3, 1:3] *= -1

        w2c = torch.inverse(c2w)
        return w2c


    def __init__(self, cfg, colors=None, poses=None, frame_ids=None, K=None, depths=None, masks=None, point_cloud=None, wandb_run=None, *args, **kwargs):
        # load cfg
        self.cfg = cfg

        # Get the current date and time
        current_datetime = datetime.datetime.now()
        
        # Format the date and time as a string
        formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        cfg['train']['save_path'] += f'/{formatted_datetime}'

        # create ConfigParams from dict
        self.opt = ConfigParams(**cfg['train'])
        self.pipe = ConfigParams(**cfg['pipeline'])

        # initialize a empty gaussian models
        self.gaussians = GaussianModel(cfg["gaussians_model"]["sh_degree"])
        self.allDatasForTrain = [] 

        self._cameras_opt_cnt = {}

        # variables
        self.debug = cfg['debug']
        self.device = cfg['device']
        self._down_scale_ratio = down_scale_ratio = int(cfg['down_scale_ratio'])

        bg_color = [1, 1, 1] if cfg['white_background'] else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device=self.device)

        # cameras offset
        self._cameras_translate = self._cameras_radius = None

        # wandb logger
        if wandb is not None:
            self.wandb_run =  wandb_run

        # initialize tqdm bar
        self._bar = None

        # load from dataset
        is_group_params = all(var in kwargs.keys() for var in ['dataset', 'opt', 'pipe'])
        if is_group_params:
            self._create_all_from_GroupParams(kwargs)
            return

        get_new_datas = all(data is not None for data in [colors, poses, frame_ids, K])
        if not get_new_datas:
            print("WARNING: No DATAs are provided. not ready for trainning")
            print("WARNING: No Gaussian Splat Model is initialized. Not Allowed create a Gaussian Model from PointCloud without image datas")
            return

        # get image size from colors
        if len(colors.shape) == 4:
            if colors.shape[1] == 3:
                image_height, image_width = colors.shape[2], colors.shape[3]

            elif colors.shape[3] == 3:
                image_height, image_width = colors.shape[1], colors.shape[2]

        else:
            raise Exception("ERROR: shape of colors is invalid. either (N, h, w, c) or (N, c, h, w)")

        self.frameCnt = colors.shape[0] 

        self.image_height = image_height
        self.image_width = image_width

        # create viewpointDict
        # images to Tensor
        colors = colors[:, ::down_scale_ratio, ::down_scale_ratio, :]
        colors = batch_images_from_numpy_to_tensor(colors).to(self.device)

        if depths is not None:
            depths = depths[:, ::down_scale_ratio, ::down_scale_ratio]
            depths = batch_images_from_numpy_to_tensor(depths).to(self.device)
        else:
            depths = [None for _ in range(self.frameCnt)]

        if masks is not None:
            masks = masks[:, ::down_scale_ratio, ::down_scale_ratio]
            masks = batch_images_from_numpy_to_tensor(masks).to(self.device)

        else:
            masks = [None for _ in range(self.frameCnt)]

        self.K = K

        datasForTrainList = []
        for (color, depth, mask, pose, frame_id) in zip(colors, depths, masks, poses, frame_ids):
            c2w_quat = matrix_to_quaternion(torch.tensor(pose[:3, :3]).float())
            c2w_t = torch.tensor(pose[:3, 3]).float()

            # TODO remove later
            c2w = pose.copy()
            c2w[:3, 1:3] *= -1
            w2c = np.linalg.inv(c2w)

            cam = self.setup_camera(w2c)

            # w2c, c2w numpy
            cur_data = {
                    'color': color,
                    'depth': depth,
                    'mask' : mask,
                    'cam'  : cam,
                    'frame_id' : frame_id,
                    'w2c' : w2c,
                    'c2w_quat': c2w_quat, 
                    'c2w_t': c2w_t,
                    'opt_cnt' : 0
                    }

            datasForTrainList.append(cur_data)

        self.allDatasForTrain += datasForTrainList

        self._update_camera_extent()
        
        msg = f"INFO: GS Initialization done. {len(self.allDatasForTrain)} new frames added, "
        # create gaussian model
        if point_cloud is not None:
            self._create_gaussian_model_from_pcd(point_cloud)
            msg += "point cloud is provided, GS starts from a point cloud"

        else:
            self._create_gaussian()
            msg += "point cloud is not provided, GS starts from scratch"

        print(msg)

    def create_camera(self, w2c, near=0.01, far=100, bg=[0, 0, 0]):
        k = self.K
        fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
        w, h = self.image_width, self.image_height

        if len(w2c.shape) == 2:
            cam_center = torch.inverse(w2c)[:3, 3]
            w2c = w2c.unsqueeze(0).transpose(1, 2)
        else:
            w2c = w2c.transpose(1, 2)
            cam_center = torch.inverse(w2c)[:, :3, 3]

        opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                    [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                    [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                    [0.0, 0.0, 1.0, 0.0]]).cuda().float().unsqueeze(0).transpose(1, 2)
        full_proj = w2c.bmm(opengl_proj)
        cam = Camera(
            image_height=h,
            image_width=w,
            tanfovx=w / (2 * fx),
            tanfovy=h / (2 * fy),
            bg=torch.tensor(bg, dtype=torch.float32, device=self.device),
            scale_modifier=1.0,
            viewmatrix=w2c,
            projmatrix=full_proj,
            sh_degree=0,
            campos=cam_center,
            prefiltered=False
        )
        return cam

    def setup_camera(self, w2c, near=0.01, far=100, bg=[0, 0, 0]):
        k = self.K
        fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
        w, h = self.image_width, self.image_height

        w2c = torch.tensor(w2c).cuda().float()
        cam_center = torch.inverse(w2c)[:3, 3]
        w2c = w2c.unsqueeze(0).transpose(1, 2)

        cam_center = torch.inverse(w2c)[:3, 3]
        opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                    [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                    [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                    [0.0, 0.0, 1.0, 0.0]]).cuda().float().unsqueeze(0).transpose(1, 2)
        full_proj = w2c.bmm(opengl_proj)
        cam = Camera(
            image_height=h,
            image_width=w,
            tanfovx=w / (2 * fx),
            tanfovy=h / (2 * fy),
            bg=torch.tensor(bg, dtype=torch.float32, device=self.device),
            scale_modifier=1.0,
            viewmatrix=w2c,
            projmatrix=full_proj,
            sh_degree=0,
            campos=cam_center,
            prefiltered=False
        )
        return cam

    def _create_all_from_GroupParams(self, kwargs):
        dataset = kwargs.pop('dataset')
        opt = kwargs.pop('opt')
        pipe = kwargs.pop('pipe')

        # create scene
        self.scene = Scene(dataset, self.gaussians)

        # overwrite config background
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device=self.device)

        # get camera stack [Camera], multi-res
        self._scene_load_viewpoints()

        if "checkpoint" in kwargs.keys():
            checkpoint = kwargs.pop('checkpoint')
            if not os.path.exists(checkpoint):
                raise Exception(f"ERROR: CHECKPOINT PATH ERROR, {checkpoint} does not exist")
            (model_params, self._gs_iter) = torch.load(checkpoint)
            self.gaussians.restore(model_params, opt)

        if "start_pcd" in kwargs.keys():
            pcd = kwargs.pop('start_pcd')
            self._create_gaussian_model_from_pcd(pcd)
        else:
            self._create_gaussian()

    @classmethod
    def from_GroupParams(cls, cfg, dataset, opt, pipe, *args, **kwargs):
        # Define a tuple containing the variables
        variables = (dataset, opt, pipe)
        
        # Check isinstance for each variable in the tuple
        is_group_params = all(isinstance(var, GroupParams) for var in variables)
        assert is_group_params, "[ERROR][gaussian_splat_runner.py] dataset, opt, pipe must be GroupParams"

        return cls(cfg=cfg, dataset=dataset, opt=opt, pipe=pipe, *args, **kwargs)

    def _scene_load_viewpoints(self):
        viewpoint_train_stack = self.scene.getTrainCameras(self._down_scale_ratio).copy()
        viewpoint_test_stack = self.scene.getTestCameras(self._down_scale_ratio).copy()

        if len(viewpoint_train_stack) == 0:
            raise Exception("[ERROR] LoadTrainViewpointsError: There no viewpoints for training") 

        # create a hashcode for camera stack
        for viewpoint in viewpoint_train_stack:
            frame_id = str(viewpoint.uid)
            hash_object = hashlib.sha256(frame_id.encode())
            hash_code = hash_object.hexdigest()
            self.allDatasForTrain[hash_code] = viewpoint

        if len(viewpoint_test_stack) > 0:
            for viewpoint in viewpoint_test_stack:
                frame_id = str(viewpoint.uid)
                hash_object = hashlib.sha256(frame_id.encode())
                hash_code = hash_object.hexdigest()
                self.viewpointsForTestDict[hash_code] = viewpoint

        self._update_camera_extent()


    def _update_camera_extent(self):
        # get nerfnormalization
        if list(self.allDatasForTrain) == 0:
            return
        else:
            camera_extent = self.getNerfppNorm([data['w2c'] for data in self.allDatasForTrain])
            self._cameras_translate = camera_extent['translate']
            self._cameras_radius = camera_extent['radius']
            return 1 

    def _create_gaussian(self):
        # initialize random gaussian in unit space, please check all parameters of GS, how to do it in unit coordinate
        # TODO config number of points for gaussian model, load input.ply, and downsample

        # load point cloud from input.ply TODO set a global gs_runner workspace
        file_path = 'input.ply'
        
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        normals = np.asarray(pcd.normals)               # check if raise error if no normals provides, gaussian model does not requires normals

        pcd = BasicPointCloud(points=points, colors=colors, normals=normals)

        if not self._update_camera_extent:
            raise Exception("ERROR: spatical_lr_scale can not be obtained from cameras, there is no camera provided")
            
        self.gaussians.create_from_pcd(pcd, self._cameras_radius)
        self.gaussians.training_setup(self.opt)

        # reset first iter
        self._gs_iter = 1 


    def _create_gaussian_model_from_pcd(self, pcd: PointCloud):
        # create gaussian splatting model from open3d.geometry.PointCloud
        if self.cfg['use_octree']:
            # TODO build a octree from pointCloud, and create a color points sampled from voxel, voxel masks
            pass

        # downsample points
        print("Downsample the point cloud with a voxel of 0.02")
        voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.02)

        # remove statistic outlier
        print("Statistical oulier removal")
        cl, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=20,           # TODO hyparameters must be adjust for unit coordinate
                                                               std_ratio=2.0)

        pcd = voxel_down_pcd.select_by_index(ind)

        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)

        if colors.shape == (0 , 3):
            #colors = np.ones(points.shape) * 0.5
            colors = np.random.rand(*points.shape)    # start from random color

        normals = np.asarray(pcd.normals)               # not required

        pcd = BasicPointCloud(points=points, colors=colors, normals=normals)

        if not self._update_camera_extent:
            raise Exception("ERROR: spatical_lr_scale can not be obtained from cameras, there is no camera provided")
            
        self.gaussians.create_from_pcd(pcd, self._cameras_radius)
        self.gaussians.training_setup(self.opt)
        
        # reset first iter
        self._gs_iter = 1 

    def load_gaussian_from_ply(self, plyfn):
        assert os.path.isfile(plyfn), "There is no input.ply"
        try:
            self.gaussians.load_ply(plyfn)
            self.gaussians.training_setup(self.opt)
        except Exception as e:
            print("WARNING: Gaussian Model Load from PLY failed. Start to create a basis Gaussian Model")
            self._create_gaussian()

    def add_new_frames(self, colors, poses, frame_ids, depths=None, masks=None, new_pcd=None, reuse_gaussian=True):
        get_new_datas = all(data is not None for data in [colors, poses, frame_ids])
        if not get_new_datas:
            print("WARNING: No New DATAs are provided. colors, poses, and frame_ids must be provided")
            return

        down_scale_ratio = self._down_scale_ratio
        newImagesCnt = colors.shape[0] 

        # create viewpointDict
        # images to Tensor
        colors = colors[:, ::down_scale_ratio, ::down_scale_ratio, :]
        colors = batch_images_from_numpy_to_tensor(colors).to(self.device)

        if depths is not None:
            depths = depths[:, ::down_scale_ratio, ::down_scale_ratio]
            depths = batch_images_from_numpy_to_tensor(depths).to(self.device)

        else:
            depths = [None for _ in range(self.frameCnt)]

        if masks is not None:
            masks = masks[:, ::down_scale_ratio, ::down_scale_ratio]
            masks = batch_images_from_numpy_to_tensor(masks).to(self.device)
        else:
            masks = [None for _ in range(self.frameCnt)]

        if self.K is None: 
            print("ERROR: Camera intrinsic has not been set up, add_new_frames failed.")
            return

        datasForTrainList = []
        for (color, depth, mask, pose, frame_id) in zip(colors, depths, masks, poses, frame_ids):
            c2w_quat = matrix_to_quaternion(torch.tensor(pose[:3, :3]).float())
            c2w_t = torch.tensor(pose[:3, 3]).float()

            # TODO remove later
            c2w = pose.copy()
            c2w[:3, 1:3] *= -1
            w2c = np.linalg.inv(c2w)

            cam = self.setup_camera(w2c)

            # w2c, c2w numpy
            cur_data = {
                    'color': color,
                    'depth': depth,
                    'mask' : mask,
                    'cam'  : cam,
                    'frame_id' : frame_id,
                    'w2c' : w2c,
                    'c2w_quat': c2w_quat, 
                    'c2w_t': c2w_t,
                    'opt_cnt' : 0
                    }
            datasForTrainList.append(cur_data)

        self.allDatasForTrain += datasForTrainList

        self._update_camera_params(datasForTrainList)

        self._update_camera_extent()

        msg = f"INFO: {newImagesCnt} new frames added, total {len(self.allDatasForTrain)} frames"

        if not reuse_gaussian:
            if new_pcd is None:
                self._create_gaussian()
                msg += ", restart training"
            else:
                self._create_gaussian_model_from_pcd(new_pcd)
                msg += ", restart training from provided point clouds"
        else:
            msg += " Reusing the GS, continue training"

        print(msg)

    def train(self, once_iterations=100_000):
        ema_loss_for_log = 0.0

        # TODO iter time: gaussian iteration, camera iterations
        # TODO train should start from gaussian optimization and after try to optimize camera

        # create camera optimizer, optimize all camera frame
        """
        #camera_matrix is not differentiable through backward GaussianRasterization

        c2w_quats = [data['c2w_quat'] for data in self.allDatasForTrain]
        c2w_ts = [data['c2w_t'] for data in self.allDatasForTrain]

        c2w_quats = torch.stack(c2w_quats, axis=0).float().to(self.device)
        c2w_ts = torch.stack(c2w_ts, axis=0).float().to(self.device)
        
        #c2w_quats.requires_grad_(True)
        #c2w_ts.requires_grad_(True)

        camParams = [
            {'params': [c2w_ts], 'lr': 0.01, "name": "c2w_t"},
            {'params': [c2w_quats], 'lr': 0.01, "name": "c2w_quats"}
            ] 

        cam_optimizer = torch.optim.Adam(camParams, lr=0.0, eps=1e-15)

        w2cAll = self.c2w_to_w2c(c2w_quats, c2w_ts)
        """

        if self._bar is None:
            self._bar = tqdm(range(self._gs_iter, self.opt.iterations+1), desc="Training progress")

        for _ in range(once_iterations):
            if self._gs_iter == self.opt.iterations:
                print(f"INFO: Training completely finished after {self.opt.iterations} iterations")
                return

            self.gaussians.update_learning_rate(self._gs_iter)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if self._gs_iter % 1000 == 0:
                self.gaussians.oneupSHdegree()

            # Pick a random Camera
            #TODO should remove
            if isinstance(self.allDatasForTrain, dict):
                key = random.choice(list(self.allDatasForTrain.keys()))
                viewpoint_cam = self.allDatasForTrain[key] 
            
                bg = torch.rand((3), device=self.device) if self.opt.random_background else self.background

                render_pkg = render(viewpoint_cam, self.gaussians, self.pipe, bg)
                image, viewspace_point_tensor, visibility_filter, radii, depth = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["depth"]

                # Loss
                gt_image = viewpoint_cam.original_image.cuda()

                try:
                    mask = viewpoint_cam.mask.bool().cuda()
                    depth_gt = viewpoint_cam.depth.cuda()
                except:
                    mask = None
                    depth_gt = None

            else:
                idx = random.randint(0, len(self.allDatasForTrain)-1)
                data = self.allDatasForTrain[idx]
                gt_image = data['color'].cuda()
                mask = data['mask'].bool() if data['mask'] is not None else None
                depth_gt = data['depth'].cuda() if data['depth'] is not None else None

                cam = data['cam']

                 
                #self.gaussians.deactivate_grad()
                render_pkg = renderer(cam, self.gaussians, self.pipe)
                image, viewspace_point_tensor, visibility_filter, radii, depth = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["depth"]

            Ll2 = 0.

            if mask is not None:
                Ll1 = l1_loss(torch.masked_select(image, mask), torch.masked_select(gt_image, mask))
                # add valid depth in mask
                if depth_gt is not None:
                    Ll2 = l2_loss(torch.masked_select(depth, mask), torch.masked_select(depth_gt, mask))
            else:
                Ll1 = l1_loss(image, gt_image)
                if depth_gt is not None:
                    Ll2 = l2_loss(depth, depth_gt)

            Lssim = 1.0 - ssim(image, gt_image)
            loss = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * Lssim + 1e-2 * Ll2
            
            wandb.log({"total": loss, "l1": Ll1, "d-simm": Lssim, 'depth_l2': 1e-2 * Ll2})

            # TODO backward every 10 frames
            loss.backward()
           
            if self.debug:
                if self._gs_iter % 100 == 0:
                    fig = plt.figure(figsize=(20,10))
                    plt.subplot(1, 2, 1); plt.imshow(depth.detach().cpu().squeeze().numpy()); plt.colorbar(); plt.axis('off'); plt.title('Rendered Depth')
                    plt.subplot(1, 2, 2); plt.imshow(depth_gt.detach().cpu().squeeze().numpy()-depth.detach().cpu().squeeze().numpy()); plt.colorbar(); plt.axis('off'); plt.title('Depth diff')
                    data = get_img_from_fig(fig)

                    # avoid memory leak
                    plt.close()

                    gt_image_np = gt_image.detach().cpu().numpy().transpose(1, 2, 0)
                    image_np    = image.detach().cpu().numpy().transpose(1, 2, 0)
                    gt_image_np = (gt_image_np * 255).astype(np.uint8)
                    image_np = (image_np * 255).astype(np.uint8)

                    images = np.hstack((gt_image_np, image_np))
                    images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
                    size = images.shape[:2][::-1]
                    resized_depths = cv2.resize(data, size)
                    images = np.vstack((images, resized_depths))

                    # add iter num on the corner
                    # Add text to the image
                    text = f"Iteration: {self._gs_iter} Loss: {float(loss):.6f}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1
                    font_thickness = 2
                    text_color = (255, 255, 255)  # White color
                    text_position = (50, 50)  

                    cv2.putText(images, text, text_position, font, font_scale, text_color, font_thickness)
                    # live show
                    if False:
                        cv2.imshow("Diff", images)

                        # Check for key press
                        key = cv2.waitKey(30) & 0xFF
                        if key == ord('q'):  # Press 'q' to quit
                            break
                      
                    else:
                        save_path = 'debug_images'
                        if not os.path.exists(save_path):
                            # Create the directory
                            os.mkdir(save_path)
                            print("Directory created successfully.")

                        fn = f"debug_images/image_iter_{self._gs_iter:06d}.jpg"
                        cv2.imwrite(fn, images)



            del depth, depth_gt, gt_image, image, mask
            torch.cuda.empty_cache()

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if self._gs_iter % 10 == 0:
                    self._bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    self._bar.update(10)
                if self._gs_iter == self.opt.iterations:
                    self._bar.close()

                # save
                if (self._gs_iter in self.opt.save_pcd_iterations):
                   print("\n[ITER {}] Saving Gaussians".format(self._gs_iter))
                   point_cloud_path = os.path.join(self.opt.save_path, "point_cloud/iteration_{}".format(self._gs_iter))
                   self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

                # Densification
                if self._gs_iter < self.opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    # TODO computation of scene.cameras_extent is not trival
                    if self._gs_iter > self.opt.densify_from_iter and self._gs_iter % self.opt.densification_interval == 0:
                        size_threshold = 20 if self._gs_iter > self.opt.opacity_reset_interval else None
                        self.gaussians.densify_and_prune(self.opt.densify_grad_threshold, 0.005, self._cameras_radius, size_threshold)
                    if self._gs_iter % self.opt.opacity_reset_interval == 0 or (self.cfg['white_background'] and self._gs_iter == self.opt.densify_from_iter):
                        self.gaussians.reset_opacity()

                # Optimizer step
                if self._gs_iter < self.opt.iterations:
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none = True)

                if (self._gs_iter in self.opt.save_ck_iterations):
                    print("\n[ITER {}] Saving Checkpoint".format(self._gs_iter))
                    torch.save((self.gaussians.capture(), self._gs_iter), self.opt.save_path + "/checkpoint_" + str(self._gs_iter) + ".pth")

            self._gs_iter += 1


    


if __name__ == '__main__':
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1_00, 5_00, 1_00, 3_000, 15_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    # start a new experiment
    dataset = lp.extract(args).source_path.split('/')[-1]
    wandb.login()
    run = wandb.init(
        # Set the project where this run will be logged
        project="Gaussian Splatting Models",
        name="naive-GS-{dataset}",
        # Track hyperparameters and run metadata
        settings=wandb.Settings(start_method="fork"),
        mode='disabled'
    )

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    cfg = yaml.safe_load(open('gs_runner_config.yml', 'r'))

    if True:
        # add gaussian noise on start point cloud
        start_pcd_fn = '/home/yjin/repos/gaussian-splatting/output/point_cloud/iteration_1000/point_cloud.ply'
        # Load the point cloud
        voxel_size = 0.1
        point_cloud = o3d.io.read_point_cloud(start_pcd_fn)

        N = 10_000
        points = np.asarray(point_cloud.points)
        colors = np.asarray(point_cloud.points)

        repeat_N = 2
        points = np.repeat(points, repeat_N, axis=0)
        colors = np.repeat(colors, repeat_N, axis=0)

        points += (np.random.rand(*points.shape)-0.5) * voxel_size * 3

        big_pcd = o3d.geometry.PointCloud()
        big_pcd.points = o3d.utility.Vector3dVector(points)
        #big_pcd.colors = o3d.utility.Vector3dVector(colors)

    # start a new experiment
    dataset = lp.extract(args).source_path.split('/')[-1]

    wandb.login()
    run = wandb.init(
        # Set the project where this run will be logged
        project="Gaussian Splatting Models",
        name="naive-GS-{dataset}",
        # Track hyperparameters and run metadata
        settings=wandb.Settings(start_method="fork"),
        #mode='disabled'
    )



    gsRunner = GaussianSplatRunner.from_GroupParams(cfg, lp.extract(args), op.extract(args), pp.extract(args))
    gsRunner.train()



