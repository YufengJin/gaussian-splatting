import os
import sys
import numpy as np
import yaml
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
from gs_runner import GaussianSplatRunner

# load all .npy
folder_path = '/home/yjin/repos/BundleSDF/final_datas' 

# List all .npy files in the folder
npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]

# Loop through each .npy file
for file_name in npy_files:
    # Load the NumPy array from the .npy file
    arr = np.load(os.path.join(folder_path, file_name))

    # Assign the filename as the variable name for the array
    var_name = os.path.splitext(file_name)[0]  # Remove the .npy extension
    globals()[var_name] = arr

# Now you can access the loaded arrays using their filenames as variable names
# For example, if you have a file named "example.npy", you can access its array using "example" variable

# create a transform from gl cam to gs cams
cfg = yaml.safe_load(open('gs_runner_config.yml', 'r'))

import wandb
wandb.login()
run = wandb.init(
    # Set the project where this run will be logged
    project="Gaussian Splatting Analysis",
    name="naive-GS-bundlesdf-datasets-new-renderer",
    # Track hyperparameters and run metadata
    settings=wandb.Settings(start_method="fork"),
    mode='disabled'
)


frameIds = [f'{i:03d}' for i in range(rgbs.shape[0])]

def rgbd_to_pointcloud(rgb_image, depth_image, K):
    fx = K[0][0]
    fy = K[1][1]
    cx = K[0][2]
    cy = K[1][2]

    h, w = depth_image.shape
    y, x = np.indices((h, w))
    z = depth_image
    x = (x - cx) * z / fx
    y = (y - cy) * z / fy

    # Stack the coordinates to create the point cloud
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    
    # Step 4: Associate colors with the point cloud
    colors = rgb_image.reshape(-1, 3)
    # Step 5: Create Open3D point cloud and visualize
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Normalize colors to range [0, 1]

    return pcd

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

def preprocess_datas(rgbs, depths, masks, glcam_in_obs):
    pcdAll = o3d.geometry.PointCloud()
    for (color, depth, mask, pose) in zip(rgbs, depths, masks, glcam_in_obs):
        # mask erosion
        mask_uint = mask.astype(np.uint8)
        kernel = np.ones((30,30), np.uint8)  # You can adjust the kernel size as needed
    
        # Perform erosion
        eroded_mask = cv2.erode(mask_uint, kernel, iterations=1)
    
        mask = eroded_mask.astype(np.bool_)
    
        depth_tmp = depth.copy()
        depth_tmp[depth_tmp < 0.1] = np.nan
    
        mask = np.logical_and(np.logical_not(np.isnan(depth_tmp)), mask)
        #plt.subplot(1, 3, 1); plt.imshow(mask);
        #plt.subplot(1, 3, 2); plt.imshow(mask_uint.astype(np.bool_));
        #plt.subplot(1, 3, 3); plt.imshow(mask ^ mask_uint.astype(np.bool_));
        #plt.show()
    
        # mask
        color[np.logical_not(mask)] = [0., 0., 0.]
        depth[np.logical_not(mask)] = 0.
        pcd = rgbd_to_pointcloud(color, depth, K)
    
        pose_o3d = pose.copy()
        pose_o3d[:3, 1:3] *= -1
        pcd.transform(pose_o3d)
        pcdAll += pcd
    
        # postprocess pointlcoud remove outlier
        #pcdAll = pcdAll.voxel_down_sample(voxel_size=0.01)
        #cl, ind = pcdAll.remove_statistical_outlier(nb_neighbors=20,
        #                                                std_ratio=1.0)
    
        #pcdAll = pcdAll.select_by_index(ind)
    
    
    pcdAll = pcdAll.voxel_down_sample(voxel_size=0.01)
    #cl, ind = pcdAll.remove_statistical_outlier(nb_neighbors=100,
    #                                                std_ratio=1.0)
    cl, ind = pcdAll.remove_radius_outlier(nb_points=20, radius=0.02)

    display_inlier_outlier(pcdAll, ind)
    
    pcdAll = pcdAll.select_by_index(ind)
    #o3d.visualization.draw_geometries([pcdAll])
    return pcdAll




pcdAll = preprocess_datas(rgbs, depths, masks, glcam_in_obs)

#gsRunner = GaussianSplatRunner(cfg, colors=rgbs, depths=depths, masks=masks, poses=glcam_in_obs, frame_ids=frameIds, K=K, point_cloud=pcdAll, wandb=run)
gsRunner = GaussianSplatRunner(cfg, colors=rgbs, depths=depths, masks=masks, poses=glcam_in_obs, frame_ids=frameIds, K=K, wandb=run)
gsRunner.train()


