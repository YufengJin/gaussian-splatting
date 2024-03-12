import os
from PIL import Image
import shutil

def copy_file(source_path, destination_path):
    # Copy the file from source_path to destination_path
    shutil.copy2(source_path, destination_path)
    print(f"File '{source_path}' copied to '{destination_path}' successfully!")

def convert_png_to_jpg(png_path, jpg_path):
    # Open the PNG image
    png_image = Image.open(png_path)

    # Convert the PNG image to RGB mode (if it's not already in RGB)
    if png_image.mode != 'RGB':
        png_image = png_image.convert('RGB')

    # Save the image as JPEG
    png_image.save(jpg_path, 'JPEG')
    print(f"PNG image converted to JPEG: {jpg_path}")

def generate_rgba(rgb_path, mask_path, output_path):
    # Open RGB image
    rgb_image = Image.open(rgb_path)
    
    # Open mask image and convert to grayscale
    mask_image = Image.open(mask_path).convert("L")
    
    # Create RGBA image
    rgba_image = Image.new("RGBA", rgb_image.size)
    
    # Combine RGB channels from the original image and alpha channel from the mask
    rgba_image.paste(rgb_image, (0, 0), mask_image)
    
    # Save the resulting image
    rgba_image.save(output_path)

if __name__ == "__main__": 
    # load all rgb images
    image_root = "/home/yjin/repos/BundleSDF/milk_bottle"
    rgb_folder = os.path.join(image_root, "rgb")
    masks_folder = os.path.join(image_root, "masks")

    # create sampled data
    rgba_folder = os.path.join(image_root, "rgba_small")
    rgb_s_folder = os.path.join(image_root, "rgb_small")
    mask_s_folder = os.path.join(image_root, "masks_small")

    if not os.path.exists(rgba_folder):
        os.makedirs(rgba_folder)
        print(f"INFO: Folder '{rgba_folder}' created successfully!")
    else:
        print(f"Folder '{rgba_folder}' already exists.")

    if not os.path.exists(rgb_s_folder):
        os.makedirs(rgb_s_folder)

    if not os.path.exists(mask_s_folder):
        os.makedirs(mask_s_folder)

    rbgFileNames = os.listdir(rgb_folder)

    for i, fn in enumerate(rbgFileNames):
        # reduce amounts of images
        if not i % 5 == 0:
            continue

        # TODO: accept all extension .jpg, .png
        rgbFn = os.path.join(rgb_folder, fn)
        maskFn = os.path.join(masks_folder, fn)
        if not os.path.isfile(maskFn): 
            print(f"WARNING: there is no corresponding mask for the rgb image({fn})")
            continue

        # execute once
        rgbaFn = os.path.join(rgba_folder, fn)
        generate_rgba(rgbFn, maskFn, rgbaFn)
        print(f"INFO: ({i}/{len(rbgFileNames)})new rgba images saved on the path {rgbaFn}")
        # copy rgb to new folder
        rgbJPGFn = fn[:-4]+".jpg"

        newRGBFn = os.path.join(rgb_s_folder, rgbJPGFn)

        # convert rgb .png to .jpg
        convert_png_to_jpg(rgbFn, newRGBFn)

        # copy masks 
        newMaskFn = os.path.join(mask_s_folder, rgbJPGFn+'.png')
        copy_file(maskFn, newMaskFn)

