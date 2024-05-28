#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, se3_to_SE3, getWorld2View2_ts, image_gradient, image_gradient_mask

# Define a class named Camera_Pose. The code is based on the camera_transf class in iNeRF. You can refer to iNeRF at https://github.com/salykovaa/inerf.
class Camera_Pose(nn.Module):
    def __init__(self,start_pose_w2c, FoVx, FoVy, image_width, image_height,
             trans=torch.tensor([0.0, 0.0, 0.0]), scale=1.0,requires_grad=False
             ):
        super(Camera_Pose, self).__init__()

        self.FoVx = FoVx
        self.FoVy = FoVy

        self.image_width = image_width
        self.image_height = image_height

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale
        self.cov_offset = 0
        
        self.w = nn.Parameter(torch.normal(0., 1e-6, size=(3,)).to(start_pose_w2c.device), requires_grad=requires_grad)
        self.v = nn.Parameter(torch.normal(0., 1e-6, size=(3,)).to(start_pose_w2c.device), requires_grad=requires_grad)
        
        self.forward(start_pose_w2c)

    def forward(self, start_pose_w2c):
        deltaT=se3_to_SE3(self.w,self.v)
        self.pose_w2c = torch.matmul(deltaT, start_pose_w2c.inverse()).inverse()
        self.update()
    
    def current_campose_c2w(self):
        return self.pose_w2c.inverse().clone().cpu().detach().numpy()

    def update(self):
        self.world_view_transform = self.pose_w2c.transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

class BigCamera(Camera):
    def __init__(self,  colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask, depth, mask, frame_id,
            image_name, uid, trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"):

       super(BigCamera, self).__init__(colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid, trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda")

       self.depth = self.mask = None

       if isinstance(depth, torch.Tensor):
           self.depth = depth.to(self.data_device)

       if isinstance(mask, torch.Tensor):
           self.mask = mask.to(self.data_device)

       self.frame_id = frame_id

class MonoGSCamera(nn.Module):
    def __init__(
        self,
        uid,
        color,
        depth,
        gt_T,
        fovx,
        fovy,
        image_height,
        image_width,
        trans=np.array([0.0, 0.0, 0.0]), 
        scale=1.0,
        device="cuda:0",
    ):
        super(MonoGSCamera, self).__init__()
        self.uid = uid
        self.device = device

        T = torch.eye(4, device=device)
        self.R = T[:3, :3]
        self.T = T[:3, 3]

        if not isinstance(gt_T, torch.Tensor):
            gt_T = torch.tensor(gt_T, device=device)
        self.R_gt = gt_T[:3, :3]
        self.T_gt = gt_T[:3, 3]

        self.original_image = color.clamp(0.0, 1.0).to(device)
        self.depth = depth
        self.grad_mask = None

        self.FoVx = fovx
        self.FoVy = fovy
        self.image_height = image_height
        self.image_width = image_width

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = torch.tensor(trans, device=device)
        self.scale = torch.tensor(scale, device=device)

        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()

        self.cam_rot_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )
        self.cam_trans_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )

        self.exposure_a = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )
        self.exposure_b = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )

    @staticmethod
    def init_from_dataset(dataset, idx, projection_matrix):
        gt_color, gt_depth, gt_pose = dataset[idx]
        return Camera(
            idx,
            gt_color,
            gt_depth,
            gt_pose,
            projection_matrix,
            dataset.fx,
            dataset.fy,
            dataset.cx,
            dataset.cy,
            dataset.fovx,
            dataset.fovy,
            dataset.height,
            dataset.width,
            device=dataset.device,
        )

    @property
    def world_view_transform(self):
        return getWorld2View2_ts(self.R, self.T, self.trans, self.scale).transpose(0, 1).cuda()

    @property
    def full_proj_transform(self):
        return (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)

    @property
    def camera_center(self):
        return self.world_view_transform.inverse()[3, :3]

    def update_RT(self, R, t):
        self.R = R.to(self.device)
        self.T = t.to(self.device)

    def compute_grad_mask(self, edge_threshold=1.1):
        gray_img = self.original_image.mean(dim=0, keepdim=True)
        gray_grad_v, gray_grad_h = image_gradient(gray_img)
        mask_v, mask_h = image_gradient_mask(gray_img)
        gray_grad_v = gray_grad_v * mask_v
        gray_grad_h = gray_grad_h * mask_h
        img_grad_intensity = torch.sqrt(gray_grad_v**2 + gray_grad_h**2)

        median_img_grad_intensity = img_grad_intensity.median()
        self.grad_mask = (
            img_grad_intensity > median_img_grad_intensity * edge_threshold
        )

    def clean(self):
        self.original_image = None
        self.depth = None
        self.grad_mask = None

        self.cam_rot_delta = None
        self.cam_trans_delta = None

        self.exposure_a = None
        self.exposure_b = None
