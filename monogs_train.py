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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui, monogs_render
from scene.cameras import MonoGSCamera
import sys
import cv2
import numpy as np
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.pose_utils import update_pose
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import wandb
import kornia as ki
# matplotlib visualizations
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import Normalize
import open3d as o3d
import io
# start a new experiment
#wandb.init(project="gaussian-splatting-model")

def compute_edge_loss(image, gt_image):
    assert isinstance(image, torch.Tensor) and isinstance(gt_image, torch.Tensor)

    gt_gray = ki.color.rgb_to_grayscale(gt_image.unsqueeze(0))
    gray = ki.color.rgb_to_grayscale(image.unsqueeze(0))

    t = ki.filters.sobel(gt_gray).squeeze(0)
    e = ki.filters.sobel(gray).squeeze(0)

    return torch.abs(t - e).mean(), t, e

# define a function which returns an image as numpy array from figure
def get_img_from_fig(fig, dpi=180, BGR2RGB=False):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    if BGR2RGB:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def from_Cam_to_MonoGSCam(cam, noised=False):
    idx = cam.uid
    gt_color = cam.original_image
    gt_depth = None
    fovx = cam.FoVx
    fovy = cam.FoVy
    height = cam.image_height
    width = cam.image_width
    device = cam.data_device
    R = cam.R
    T = cam.T

    gt_pose = np.eye(4)
    gt_pose[:3, :3] = R
    gt_pose[:3, 3] = T

    #trans = np.array([0.0, 0.0, 0.5])
    #scale = 2.0
    if noised:
        T += (np.random.rand(3) - 0.5) * 2 * 0.3   # translation noise (-0.02, 0.02)
        R = R @ cv2.Rodrigues((np.random.rand(3) - 0.5) * 2 * 5 / 180 * np.pi)[0] 

    # R, T to torch tensors
    R = torch.tensor(R, device=device)
    T = torch.tensor(T, device=device)
    
    cam = MonoGSCamera(
            idx,
            gt_color,
            gt_depth,
            gt_pose,
            fovx,
            fovy,
            height,
            width,
            device=device,
        )

    cam.update_RT(R, T)
    return cam
    
def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        viewpoint = from_Cam_to_MonoGSCam(viewpoint_cam, noised=True)
        viewpoint_raw = from_Cam_to_MonoGSCam(viewpoint_cam, noised=False)


        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        with torch.no_grad():
            render_pkg = monogs_render(viewpoint_raw, gaussians, pipe, bg)
            depth_gt = render_pkg["depth"]

        opt_params = []
        opt_params.append(
            {
                "params": [viewpoint.cam_rot_delta],
                "lr": 0.01, 
                "name": "rot_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.cam_trans_delta],
                "lr": 0.001,
                "name": "trans_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.exposure_a],
                "lr": 0.01,
                "name": "exposure_a_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.exposure_b],
                "lr": 0.01,
                "name": "exposure_b_{}".format(viewpoint.uid),
            }
        )

        pose_optimizer = torch.optim.Adam(opt_params)

        for pose_iter in range(80):
            render_pkg = monogs_render(viewpoint, gaussians, pipe, bg)
            image, viewspace_point_tensor, visibility_filter, radii, depth, opacity= render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["depth"], render_pkg["opacity"]

            # Loss
            viewpoint.compute_grad_mask()
            gt_image = viewpoint.original_image.cuda()

            # TODO test mask backpropagation, understand ssim loss
            #mask = gt_image.ge(1e-5)

            depth_loss = torch.nn.functional.mse_loss(depth, depth_gt)

            #Ll1 = l1_loss(torch.masked_select(image, mask), torch.masked_select(gt_image, mask))
            Ll1 = l1_loss(image, gt_image)
            Lssim = ssim(image, gt_image)

            edge_loss, t, e = compute_edge_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - Lssim) #+ 2. * depth_loss + 2. * edge_loss
            #loss = depth_loss
            msg = f"Loss: {loss.item():.6f} L1: {Ll1.item():.6f} SSIM: {Lssim.item():.6f} Depth: {depth_loss.item():.6f} Edge: {edge_loss.item():.6f}"
            print(msg)

            pose_optimizer.zero_grad()
            loss.backward()
            with torch.no_grad():
                pose_optimizer.step()
                converged = update_pose(viewpoint)
                 
                gt_R = viewpoint.R_gt
                gt_T = viewpoint.T_gt

                R = viewpoint.R
                T = viewpoint.T

                # calculate the difference between the ground truth and the estimated pose
                R_diff = torch.norm(gt_R - R)
                T_diff = torch.norm(gt_T - T)
                print(f"DEBUG: R_diff: {R_diff.item()} T_diff: {T_diff.item()}")


            if converged:
                break   

            #wandb.log({"total": loss, "l1": Ll1, "d-simm": Lssim})
            if pose_iter % 5 == 0: 
                gt_im = gt_image.detach().cpu().numpy().transpose(1, 2, 0)
                im    = image.detach().cpu().numpy().transpose(1, 2, 0)
                depth_img = depth.detach().cpu().squeeze().numpy()
                depth_gt_img = depth_gt.detach().cpu().squeeze().numpy()
                edge = e.detach().cpu().squeeze().numpy()
                edge_gt = t.detach().cpu().squeeze().numpy()

                im_diff = np.clip(np.abs(gt_im - im), 0, 1)
                depth_diff = np.abs(depth_img - depth_gt_img)
                edge_diff = np.clip(np.abs(edge - edge_gt), 0, 1)

                fig, ax = plt.subplots(3, 3, figsize=(10, 10))
                ax[0, 0].imshow(gt_im); ax[0, 0].set_title("GT")
                ax[0, 1].imshow(im); ax[0, 1].set_title("Render")
                ax[0, 2].imshow(im_diff); ax[0, 2].set_title("Diff")
                im = ax[1, 0].imshow(depth_img); ax[1, 0].set_title("Depth"); 
                # set colorbar
                fig.colorbar(im, ax=ax[1, 0], orientation='vertical')
                im = ax[1, 1].imshow(depth_gt_img); ax[1, 1].set_title("GT Depth"); 
                fig.colorbar(im, ax=ax[1, 1], orientation='vertical')
                im = ax[1, 2].imshow(depth_diff); ax[1, 2].set_title("Depth Diff"); 
                fig.colorbar(im, ax=ax[1, 2], orientation='vertical')
                ax[2, 0].imshow(edge); ax[2, 0].set_title("Edge")
                ax[2, 1].imshow(edge_gt); ax[2, 1].set_title("GT Edge")
                ax[2, 2].imshow(edge_diff); ax[2, 2].set_title("Edge Diff")
                fig.suptitle(msg)

                img = get_img_from_fig(fig, dpi=180)
                plt.close()
                
                img = img[::2, ::2, :]
                cv2.imshow("DEBUG", img)

                # Check for key press
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):  # Press 'q' to quit
                    break

        for mapping_iter in range(0):
            continue
            render_pkg = monogs_render(viewpoint, gaussians, pipe, bg)
            image, viewspace_point_tensor, visibility_filter, radii, depth, opacity= render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["depth"], render_pkg["opacity"]
        
            # Loss
            gt_image = viewpoint.original_image.cuda()

            # TODO test mask backpropagation, understand ssim loss
            #mask = gt_image.ge(1e-5)

            #Ll1 = l1_loss(torch.masked_select(image, mask), torch.masked_select(gt_image, mask))
            Ll1 = l1_loss(image, gt_image)
            Lssim = ssim(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - Lssim)

            loss.backward()

            if mapping_iter % 5 == 0: 
                gt_image_np = gt_image.detach().cpu().numpy().transpose(1, 2, 0)
                image_np    = image.detach().cpu().numpy().transpose(1, 2, 0)
                gt_image_np = (gt_image_np * 255).astype(np.uint8)
                image_np = (image_np * 255).astype(np.uint8)
                images = np.hstack((gt_image_np, image_np))
                images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)

                black = np.zeros_like(image_np)
                image_diff = np.abs(gt_image_np - image_np).astype(np.uint8)
                image_diff = np.hstack((black, image_diff))
                images = np.vstack((images, image_diff))

                # depth_img = depth.detach().cpu().squeeze().numpy()
                # # normalize depth image to 0,1
                # depth_img = depth_img - np.min(depth_img) / (np.max(depth_img) - np.min(depth_img))
                # color_map = plt.get_cmap('viridis')  # You can change 'viridis' to other color maps
                # depth_img = color_map(depth_img)[..., :3]

                # opacity_img = opacity.detach().cpu().squeeze().numpy()
                # # normalize opacity image to 0,1
                # opacity_img = opacity_img - np.min(opacity_img) / (np.max(opacity_img) - np.min(opacity_img))
                # color_map = plt.get_cmap('viridis')  # You can change 'viridis' to other color maps
                # opacity_img = color_map(opacity_img)[..., :3]

                # depth_opa_imgs = np.hstack((depth_img, opacity_img))
                # depth_opa_imgs = (depth_opa_imgs * 255).astype(np.uint8)
                # images = np.vstack((images, depth_opa_imgs))

                text = f"Mapping Iteration: {iteration} Loss: {loss.item():.6f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                font_thickness = 2
                text_color = (255, 255, 255)  # White color
                text_position = (50, 50)

                cv2.putText(images, text, text_position, font, font_scale, text_color, font_thickness)

                images = images[::2, ::2, :]
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
                if iteration == opt.iterations:
                    progress_bar.close()

                # Log and save
                #training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
                if (iteration in saving_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)

                # Densification
                if iteration < opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

                # Optimizer step
                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

                if (iteration in checkpoint_iterations):
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[3_000, 15_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[3_000, 15_000, 30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
