import os
import sys
import numpy as np
import open3d as o3d
import json
from PIL import Image

def is_valid_se3(matrix, tolerance=1e-6):
    if not isinstance(matrix, np.ndarray) or matrix.shape != (4, 4):
        return False

    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]

    # Check if the rotation matrix is orthogonal
    if not np.allclose(np.dot(rotation.T, rotation), np.eye(3), atol=tolerance):
        return False

    # Check if the determinant of the rotation matrix is approximately 1
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=tolerance):
        return False

    return True

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


# visualize 3d coordinate frame
frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.)

camFrames =  o3d.geometry.TriangleMesh()
glCamFrames = o3d.geometry.TriangleMesh()
glCamNewFrames = o3d.geometry.TriangleMesh()
nerfCamFrames = o3d.geometry.TriangleMesh()


def from_nerf_to_gs(pose):
    mat = pose.copy()
    transform = np.eye(4)
    transform[:2, :2] = np.array([[0., -1.], [1. , 0.]])
    return mat@transform

for pose in poses:
    if is_valid_se3(pose):
        cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.1)
        pose = from_nerf_to_gs(pose)
        cam_frame.transform(pose)

        camFrames += cam_frame

for pose in glcam_in_obs:
    if np.random.rand() < 0.3:
        continue
    if is_valid_se3(pose):
        cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.1)
        cam_frame.transform(pose)
        glCamFrames += cam_frame

        # update camera pose
        pose = from_nerf_to_gs(pose)
        cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.1)
        cam_frame.transform(pose)
        glCamNewFrames += cam_frame

# load nerf camera frames
nerfDataRoot = "/home/datasets/nerf/blender/lego/"
nerfDictFn = nerfDataRoot + "/transforms_train.json"


with open(nerfDictFn, 'r') as f:
    nerfDict = json.load(f)

nerfFrames = nerfDict['frames']

for nerf_frame in nerfFrames:
    pose = np.asarray(nerf_frame['transform_matrix'])
    if not is_valid_se3(pose):
        continue

    #imgFnShort = nerfDataRoot + nerf_frame['file_path']
    #imgFn = None

    #for ext in ['.jpg', '.png']:
    #    if os.path.exists(imgFnShort + ext):
    #        imgFn = imgFnShort + ext

    #if imgFn is not None:
    #    image = Image.open(imgFn)
    #    image.show()

    cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.1)
    print(f'INFO: nerf_frame: pose: \n{pose}\n')
    cam_frame.transform(pose)

    nerfCamFrames += cam_frame


# load from bop dataset
dataRootDir = '/home/yjin/repos/BlenderProc/examples/datasets/bop_object_pose_sampling/output/bop_data/ycbv/train_pbr/000000'

cameraInfo = json.load(open(dataRootDir + '/scene_camera.json', 'r'))
cameraPoses = json.load(open(dataRootDir + '/scene_gt.json', 'r'))
camPoses = []

bopCamFrames = o3d.geometry.TriangleMesh()

for key, content in cameraPoses.items():
    cam_pose = np.eye(4)
    R = np.array(content[0]['cam_R_m2c']).reshape(3,3)
    t = np.array(content[0]['cam_t_m2c'])
    cam_pose[:3, :3] = R
    cam_pose[:3, 3] = t
    cam_pose = np.linalg.inv(cam_pose)
    cam_pose[:3, 1:3] = -cam_pose[:3,1:3]
    camPoses.append(cam_pose)


for pose in camPoses:
    if not is_valid_se3(pose):
        continue

    pose[:3, 3] /= 1000.
    cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.1)
    cam_frame.transform(pose)
    print(f'INFO: bop pose: \n{pose}\n')
    bopCamFrames += cam_frame




o3d.visualization.draw([frame, nerfCamFrames, glCamFrames, camFrames, bopCamFrames])







