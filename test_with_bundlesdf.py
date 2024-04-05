import os
import sys
import numpy as np
import yaml
from gs_runner import GaussianSplatRunner

def from_nerf_to_gs(pose):
    mat = pose.copy()
    transform = np.eye(4)
    transform[:2, :2] = np.array([[0, -1.], [1. , 0.]])
    return mat@transform

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

#TODO  zip(rgbs, depth, masks, poses, )
#TODO write a gaussian wrapper accept numpy
# create a transform from gl cam to gs cams
cfg = yaml.safe_load(open('gs_runner_config.yml', 'r'))

frameIds = [f'{i:03d}' for i in range(rgbs.shape[0])]
poses = np.asarray([from_nerf_to_gs(pose) for pose in glcam_in_obs])

gsRunner = GaussianSplatRunner(cfg, colors=rgbs, depths=depths, masks=masks, poses=glcam_in_obs, frame_ids=frameIds, K=K)

gsRunner.train()


