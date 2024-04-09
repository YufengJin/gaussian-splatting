import os
import sys
import numpy as np
import yaml
import json
import imageio
from gs_runner import GaussianSplatRunner
from PIL import Image
import matplotlib.pyplot as plt

# load from bop dataset
dataRootDir = './bop_outputs/bop_data/ycbv/train_pbr/000000'

cameraInfo = json.load(open(dataRootDir + '/scene_camera.json', 'r'))

# camera K is consistent
K = np.asarray(cameraInfo['0']['cam_K'], dtype=np.float64).reshape(3, 3)

cameraPoses = json.load(open(dataRootDir + '/scene_gt.json', 'r'))

def center_scale_camer_poses(cam_poses):
    # TODO from pointcloud
    center = np.mean(cam_poses[:, :3, 3], axis = 0)
    scale = 1.5/0.3
    cam_poses[:, :3, 3] = (cam_poses[:, :3, 3] - center) * scale
    return cam_poses

camPoses = []
rgbs = []
depths = []
masks = []
frameIds = []
for key, content in cameraPoses.items():
    # only consider one object
    imgId = int(key)
    frameIds.append(f"{imgId:06d}")

    cam_pose = np.eye(4)
    R = np.array(content[0]['cam_R_m2c']).reshape(3,3)
    t = np.array(content[0]['cam_t_m2c'])
    cam_pose[:3, :3] = R
    cam_pose[:3, 3] = t
    cam_pose = np.linalg.inv(cam_pose)
    cam_pose[:3, 1:3] = -cam_pose[:3,1:3]

    cam_pose[:3, 3] /= 1000.
    camPoses.append(cam_pose)


    fn = os.path.join(dataRootDir, 'depth', f'{imgId:06d}.png')
    depth = imageio.imread(fn)
    depth[depth == 65535] = 0
    depths.append(np.asarray(depth, dtype=np.float64)/1000.)

    fn = os.path.join(dataRootDir, 'mask', f'{imgId:06d}_000000.png')
    mask = Image.open(fn)
    masks.append(np.asarray(mask))

    # load rgb, depth
    fn = os.path.join(dataRootDir, 'rgb', f'{imgId:06d}.jpg')
    color = np.asarray(Image.open(fn), dtype=np.uint8).copy()

    color[np.logical_not(mask)] = [0, 0, 0]
    rgbs.append(color)


camPoses = np.asarray(camPoses)
rgbs = np.asarray(rgbs)
depths = np.asarray(depths)
masks = np.asarray(masks)

# create initial pointcloud from RGBD
#for (color, depth, mask) in zip(rgbs, depths, masks):
#    fig = plt.figure(figsize=(20, 10), dpi=200)
#    plt.subplot(1,3,1); plt.imshow(color)
#    plt.subplot(1,3,2); plt.imshow(depth); plt.colorbar()
#    plt.subplot(1,3,3); plt.imshow(mask)
#    plt.show()


# initialize GaussianSplatRunner
cfg = yaml.safe_load(open('gs_runner_config.yml', 'r'))

import wandb

if cfg['use_wandb']:
    wand_mode = 'online'
else:
    wand_mode = 'disabled'
wandb.login()
run = wandb.init(
    # Set the project where this run will be logged
    project="Gaussian Splatting Analysis",
    name="naive-GS-bop-datasets-new-render",
    # Track hyperparameters and run metadata
    settings=wandb.Settings(start_method="fork"),
    mode=wand_mode
)

# seperate date into multi cluster
# random select a=20 image for training

idxs = np.random.permutation(camPoses.shape[0])[:]
rgbs_init = rgbs[idxs, ...]
depths_init = depths[idxs, ...]
masks_init = masks[idxs, ...]
camPoses_init = camPoses[idxs, ...]
frameIds_init = [frameIds[ids] for ids in idxs]

gsRunner = GaussianSplatRunner(cfg, colors=rgbs_init, poses=camPoses_init, frame_ids=frameIds_init, K=K,depths= depths_init, masks=masks_init, wandb=run)


gsRunner.train()
#for _ in range(100):
#    idxs = np.random.permutation(camPoses.shape[0])[:20]
#    rgbs_tmp = rgbs[idxs, ...]
#    depths_tmp = depths[idxs, ...]
#    masks_tmp = masks[idxs, ...]
#    camPoses_tmp = camPoses[idxs, ...]
#    frameIds_tmp = [frameIds[ids] for ids in idxs]
#
#    gsRunner.add_new_frames(colors = rgbs_tmp, poses=camPoses_tmp, frame_ids=frameIds_tmp, depths=depths_tmp, masks=masks_tmp)
#    
#    gsRunner.train()


