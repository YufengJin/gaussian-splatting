import os
import numpy as np
import random
from PIL import Image

def patch_image(im_path):
    # Load the original image
    original_image = Image.open(im_path)
    
    # Convert the PIL image to a numpy array
    original_image_array = np.array(original_image)
    
    # Define the size of the patch
    patch_size = (400, 400)
    
    # Generate a random patch mask
    random_patch_mask = np.zeros_like(original_image_array, dtype=np.uint8)
    start_x = random.randint(0, original_image_array.shape[1] - patch_size[1])
    start_y = random.randint(0, original_image_array.shape[0] - patch_size[0])
    end_x = start_x + patch_size[1]
    end_y = start_y + patch_size[0]
    random_patch_mask[start_y:end_y, start_x:end_x] = 255
    
    # Apply the mask to the original image
    masked_image_array = np.copy(original_image_array)
    masked_image_array[random_patch_mask == 0] = 0
    
    # Convert the numpy array back to a PIL image
    masked_image = Image.fromarray(masked_image_array)

    # overwrite original image
    masked_image.save(im_path)

im_dir = "/home/yjin/datasets/nerf/blender/lego_occ/train"

for fn in os.listdir(im_dir):
    fulln = os.path.join(im_dir,fn)
    patch_image(fulln)
