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
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel, BigCamera
from utils.general_utils import safe_state, convert_depth_to_rgb, batch_images_from_numpy_to_tensor, image_from_numpy_to_tensor
from tqdm import tqdm
from utils.camera_utils import calculate_fov_from_K
from utils.image_utils import psnr
from utils.graphics_utils import BasicPointCloud, getWorld2View2
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, GroupParams
from open3d.geometry import PointCloud
from pdb import set_trace
def get_camTransfrom_from_glpose(pose):
    c2w = pose.copy()
    c2w[:3, 1:3] *= -1
    w2c = np.linalg.inv(c2w)
    return np.transpose(w2c[:3,:3]), w2c[:3, 3]

# TODO learning gaussian in unit coordinate or world coordiate, test on blender data
class ConfigParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        
class GaussianSplatRunner:
    @classmethod
    def getNerfppNorm(cls, cam_info):
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

        for cam in cam_info:
            W2C = getWorld2View2(cam.R, cam.T)
            C2W = np.linalg.inv(W2C)
            cam_centers.append(C2W[:3, 3:4])

        center, diagonal = get_center_and_diag(cam_centers)
        radius = diagonal * 1.1

        translate = -center

        return {"translate": translate, "radius": radius}


    def __init__(self, cfg, colors=None, depths=None, masks=None, poses=None, frame_ids=None, K=None, point_cloud=None, wandb_run=None, *args, **kwargs):
        # load cfg
        self.cfg = cfg

        # create ConfigParams from dict
        self.opt = ConfigParams(**cfg['train'])
        self.pipe = ConfigParams(**cfg['pipeline'])

        # initialize a empty gaussian models
        self.gaussians = GaussianModel(cfg["gaussians_model"]["sh_degree"])
        self.viewpointsForTrainDict = {}
        self.viewpointsForEvalDict = {}

        # variables
        self.debug = cfg['debug']
        self.device = cfg['device']
        self._down_scale_ratio = down_scale_ratio = int(cfg['down_scale_ratio'])

        bg_color = [1, 1, 1] if cfg['white_background'] else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # cameras offset
        self._cameras_translate = self._cameras_radius = None
        self._first_iter = 0

        # wandb logger
        if wandb is not None:
            self.wandb_run =  wandb_run

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
        self.fovX, self.fovY = calculate_fov_from_K(K, image_width, image_height)

        for (color, depth, mask, pose, frame_id) in zip(colors, depths, masks, poses, frame_ids):
            R, T = get_camTransfrom_from_glpose(pose)
             
            hash_object = hashlib.sha256(str(frame_id).encode())
            key = hash_object.hexdigest()

            self.viewpointsForTrainDict[key] = BigCamera(colmap_id=None, R=R, T=T,
                  FoVx=self.fovX, FoVy=self.fovY, 
                  image=color, gt_alpha_mask=None, depth=depth, mask=mask, frame_id=str(frame_id),
                  image_name=None, uid=0, data_device=self.device)                   # uid shows how many times use for training, TODO add flag to remove bad pose

        self._update_camera_extent()
        
        # create gaussian model
        if point_cloud is not None:
            self._create_gaussian_model_from_pcd(point_cloud)

        else:
            self._create_gaussian()

    def _create_all_from_GroupParams(self, kwargs):
        dataset = kwargs.pop('dataset')
        opt = kwargs.pop('opt')
        pipe = kwargs.pop('pipe')

        # create scene
        self.scene = Scene(dataset, self.gaussians)

        # overwrite config background
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # get camera stack [Camera], multi-res
        self._scene_load_viewpoints()

        if "checkpoint" in kwargs.keys():
            checkpoint = kwargs.pop('checkpoint')
            if not os.path.exists(checkpoint):
                raise Exception(f"ERROR: CHECKPOINT PATH ERROR, {checkpoint} does not exist")
            (model_params, self._first_iter) = torch.load(checkpoint)
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
            self.viewpointsForTrainDict[hash_code] = viewpoint

        if len(viewpoint_test_stack) > 0:
            for viewpoint in viewpoint_test_stack:
                frame_id = str(viewpoint.uid)
                hash_object = hashlib.sha256(frame_id.encode())
                hash_code = hash_object.hexdigest()
                self.viewpointsForTestDict[hash_code] = viewpoint

        self._update_camera_extent()


    def _update_camera_extent(self):
        # get nerfnormalization
        if list(self.viewpointsForTrainDict.values()) == 0:
            return
        else:
            camera_extent = self.getNerfppNorm(list(self.viewpointsForTrainDict.values()))
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
        self._first_iter = 0


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
        self._first_iter = 0

    def load_gaussian_from_ply(self, plyfn):
        assert os.path.isfile(plyfn), "There is no input.ply"
        try:
            self.gaussians.load_ply(plyfn)
            self.gaussians.training_setup(self.opt)
        except Exception as e:
            print("WARNING: Gaussian Model Load from PLY failed. Start to create a basis Gaussian Model")
            self._create_gaussian()


    def add_new_frames(self, colors, poses, frame_id, depths=None, masks=None, new_pcd=None, reuse_gaussian=False):
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

        if not self.K: 
            print("ERROR: Camera intrinsic has not been set up, add_new_frames failed.")

        for (color, depth, mask, pose, frame_id) in zip(colors, depths, masks, poses, frame_ids):
            R, T = get_camTransfrom_from_glpose(pose)
            #R = pose[:3, :3]
            #T = pose[:3, 3]
             
            hash_object = hashlib.sha256(str(frame_id).encode())
            key = hash_object.hexdigest()

            self.viewpointsForTrainDict[key] = BigCamera(colmap_id=None, R=R, T=T,
                  FoVx=self.fovX, FoVy=self.fovY, 
                  image=color, gt_alpha_mask=None, depth=depth, mask=mask, frame_id=str(frame_id),
                  image_name=None, uid=0, data_device=self.device)                   # uid shows how many times use for training, TODO add flag to remove bad pose

        self._update_camera_extent()

        print(f"INFO: {newImagesCnt} new frames added, total {len(self.viewpointsForTrainDict.keys())} frames")

        if not reuse_gaussian:
            if new_pcd is None:
                self._create_gaussian()
            else:
                self._create_gaussian_model_from_pcd(new_pcd)

    def _create_optimizer(self):
        # TODO optimize camera pose and gaussian model in parallel 
        pass

    def train(self):
        iter_start = torch.cuda.Event(enable_timing = True)
        iter_end = torch.cuda.Event(enable_timing = True)
        ema_loss_for_log = 0.0
        first_iter = self._first_iter

        progress_bar = tqdm(range(first_iter, self.opt.iterations), desc="Training progress")
        first_iter += 1

        for iteration in range(first_iter, self.opt.iterations + 1):
            iter_start.record()

            self.gaussians.update_learning_rate(iteration)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                self.gaussians.oneupSHdegree()

            # Pick a random Camera
            key = random.choice(list(self.viewpointsForTrainDict.keys()))
            viewpoint_cam = self.viewpointsForTrainDict[key] 
            
            #print(f'\n//////////////////////////DEBUG: viewpoint cam //////////////////////////\n \
            #\timage fovX, and fovY: {viewpoint_cam.FoVx} {viewpoint_cam.FoVy}\n \
            #\tR : {viewpoint_cam.R}\n \
            #\tt : {viewpoint_cam.T}\n \
            #\ttrans: {viewpoint_cam.trans}, scale: {viewpoint_cam.scale} \n \
            #\tworld_view_transform: {viewpoint_cam.world_view_transform}\n \
            #\tprojection matrix: {viewpoint_cam.projection_matrix}\n \
            #/////////////////////// DEBUG /////////////////////////////\n\n')

            bg = torch.rand((3), device="cuda") if self.opt.random_background else self.background

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

            #mask = gt_image.ge(1e5)

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

            #Ll2 = l2_loss(depth, depth_gt)


            Lssim = 1.0 - ssim(image, gt_image)
            loss = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * Lssim + Ll2
            
            wandb.log({"total": loss, "l1": Ll1, "d-simm": Lssim, 'depth_l2': Ll2})

            # TODO backward every 10 frames
            loss.backward()

            if self.debug:
                import matplotlib.pyplot as plt
                if iteration % 10 == 0:
                    fig = plt.figure(figsize=(20, 10), dpi=200)

                    plt.subplot(1, 2, 1); plt.imshow(depth.detach().cpu().squeeze().numpy()); plt.colorbar()
                    plt.subplot(1, 2, 2); plt.imshow(depth_gt.detach().cpu().squeeze().numpy()); plt.colorbar()
                    
                    fig.canvas.draw()

                    # Now we can save it to a numpy array.
                    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

                    gt_image_np = gt_image.detach().cpu().numpy().transpose(1, 2, 0)
                    image_np    = image.detach().cpu().numpy().transpose(1, 2, 0)
                    gt_image_np = (gt_image_np * 255).astype(np.uint8)
                    image_np = (image_np * 255).astype(np.uint8)
                    #depth_np = (convert_depth_to_rgb(depth.detach().cpu().squeeze().numpy(), vis='cv2') * 255).astype(np.uint8)[..., :3]
                    #depth_gt_np = (convert_depth_to_rgb(depth_gt.detach().cpu().squeeze().numpy(), vis='cv2') * 255).astype(np.uint8)[..., :3]

                    #depths = np.hstack((depth_gt_np,  depth_np))
                    images = np.hstack((gt_image_np, image_np))
                    size = images.shape[:2][::-1]
                    resized_depths = cv2.resize(data, size)
                    images = np.vstack((images, resized_depths))
                    cv2.imshow("Diff", images)

                    # Check for key press
                    key = cv2.waitKey(30) & 0xFF
                    if key == ord('q'):  # Press 'q' to quit
                        break

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == self.opt.iterations:
                    progress_bar.close()

                # save
                if (iteration in self.opt.save_pcd_iterations):
                   print("\n[ITER {}] Saving Gaussians".format(iteration))
                   point_cloud_path = os.path.join(self.opt.save_path, "point_cloud/iteration_{}".format(iteration))
                   self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

                # Densification
                if iteration < self.opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    # TODO computation of scene.cameras_extent is not trival
                    if iteration > self.opt.densify_from_iter and iteration % self.opt.densification_interval == 0:
                        size_threshold = 20 if iteration > self.opt.opacity_reset_interval else None
                        self.gaussians.densify_and_prune(self.opt.densify_grad_threshold, 0.005, self._cameras_radius, size_threshold)
                    if iteration % self.opt.opacity_reset_interval == 0 or (self.cfg['white_background'] and iteration == self.opt.densify_from_iter):
                        self.gaussians.reset_opacity()

                # Optimizer step
                if iteration < self.opt.iterations:
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none = True)

                if (iteration in self.opt.save_ck_iterations):
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save((self.gaussians.capture(), iteration), self.opt.save_path + "/checkpoint_" + str(iteration) + ".pth")

    


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



