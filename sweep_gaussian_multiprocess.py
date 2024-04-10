import os
import sys
import numpy as np
import yaml
import json
import imageio
import wandb
import time
import multiprocessing
from gs_runner import GaussianSplatRunner
from PIL import Image

# preprocess bop dataset
# load from bop dataset
dataRootDir = './bop_outputs/bop_data/ycbv/train_pbr/000000'

cameraInfo = json.load(open(dataRootDir + '/scene_camera.json', 'r'))

# camera K is consistent
K = np.asarray(cameraInfo['0']['cam_K'], dtype=np.float64).reshape(3, 3)

cameraPoses = json.load(open(dataRootDir + '/scene_gt.json', 'r'))

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

# initialize GaussianSplatRunner
cfg = yaml.safe_load(open('gs_runner_config.yml', 'r'))


def train(config=None):
        with wandb.init(
        # Set the project where this run will be logged
        project="Gaussian Splatting Analysis",
        config=config
        # Track hyperparameters and run metadata
        #settings=wandb.Settings(start_method="fork"),
        ) as run:

            # If called by wandb.agent, as below,
            # this config will be set by Sweep Controller
            config = wandb.config
             
            gs_cfg = cfg.copy()
            gs_cfg["train"]["position_lr_init"] = config.position_lr_init
            gs_cfg["train"]["position_lr_final"] = config.position_lr_final
            gs_cfg["train"]["position_lr_delay_unit"] = config.position_lr_delay_unit
            gs_cfg["train"]["feature_lr"] = config.feature_lr
            gs_cfg["train"]["opacity_lr"] = config.opacity_lr
            gs_cfg["train"]["scaling_lr"] = config.scaling_lr
            gs_cfg["train"]["rotation_lr"] = config.rotation_lr
            gs_cfg["train"]["lambda_dssim"] = config.lambda_dssim
            gs_cfg["train"]["densification_interval"] = config.densification_interval
            gs_cfg["train"]["opacity_reset_interval"] = config.opacity_reset_interval
            gs_cfg["train"]["densify_from_iter"] = config.densify_from_iter
            gs_cfg["train"]["densify_until_iter"] = config.densify_until_iter


            gsRunner = GaussianSplatRunner(gs_cfg, colors=rgbs, poses=camPoses, frame_ids=frameIds, K=K,depths= depths, masks=masks, wandb=run)
            gsRunner.train()

def worker(num, sweep_id, count=1):
    wandb.agent(sweep_id, function=train, count=count)


def center_scale_camer_poses(cam_poses):
    # TODO from pointcloud
    center = np.mean(cam_poses[:, :3, 3], axis = 0)
    scale = 1.5/0.3
    cam_poses[:, :3, 3] = (cam_poses[:, :3, 3] - center) * scale
    return cam_poses

if __name__ == '__main__':
    sweep_config = {
        "method": "random",
        "metric": {"goal": "minimize", "name": "total"},
        "parameters": {
            "position_lr_init": {'distribution': 'uniform',"max": 0.001, "min": 0.00001},
            "position_lr_final": {'distribution': 'uniform', "max": 0.0001, "min": 0.000001},
            "position_lr_delay_unit": {'distribution': 'uniform', "max": 0.1, "min": 0.001},
            "feature_lr": {'distribution': 'uniform', "max": 0.1, "min": 0.001},
            "opacity_lr": {'distribution': 'uniform', "max": 0.1, "min": 0.001},
            "scaling_lr": {'distribution': 'uniform', "max": 0.1, "min": 0.001},
            "rotation_lr": {'distribution': 'uniform', "max": 0.01, "min": 0.0001},
            "lambda_dssim": {'distribution': 'uniform', "max": 1, "min": 0},
            "densification_interval": {'values': [50, 100, 200]},
            "opacity_reset_interval": {'values': [1000, 3000, 5000]},
            "densify_from_iter": {'values': [200, 500, 1000]},
            "densify_until_iter": {'values': [10000, 15000, 20000]},
        }
        }

    sweep_id = wandb.sweep(sweep_config, project='Gaussian Splatting Analysis')
    num_processes = 15  # Number of parallel processes
    each_count = 200

    # Create a list to hold the process objects
    processes = []
    multiprocessing.set_start_method('spawn') 

    for i in range(num_processes):
        p = multiprocessing.Process(target=worker, args=(i, sweep_id, each_count))
        processes.append(p)
        p.start()

    # Join all the processes
    for p in processes:
        p.join()

    print("All processes have finished.")
   
