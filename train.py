# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.

import os
import torch
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import network_gui
from gaussian_renderer import render as original_render 
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

# ==============================================================================
def weighted_edge_loss(pred, gt):
  
    pred_gray = pred.mean(dim=0, keepdim=True)
    gt_gray = gt.mean(dim=0, keepdim=True)
    
    
    kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=pred.device).unsqueeze(0).unsqueeze(0)
    kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=pred.device).unsqueeze(0).unsqueeze(0)
    
    grad_x_pred = F.conv2d(pred_gray.unsqueeze(0), kernel_x, padding=1)
    grad_y_pred = F.conv2d(pred_gray.unsqueeze(0), kernel_y, padding=1)
    mag_pred = torch.sqrt(grad_x_pred**2 + grad_y_pred**2 + 1e-6)
    
    grad_x_gt = F.conv2d(gt_gray.unsqueeze(0), kernel_x, padding=1)
    grad_y_gt = F.conv2d(gt_gray.unsqueeze(0), kernel_y, padding=1)
    mag_gt = torch.sqrt(grad_x_gt**2 + grad_y_gt**2 + 1e-6)
    
    return l1_loss(mag_pred, mag_gt)

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    if not os.path.exists(dataset.model_path):
        os.makedirs(dataset.model_path, exist_ok=True)
        
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(dataset.model_path)

    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    
    viewpoint_stack = scene.getTrainCameras().copy()
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
                    net_image = original_render(custom_cam, gaussians, pipe, background, scaling_modifier, dataset.train_test_exp, SPARSE_ADAM_AVAILABLE)["render"]
        
                    vis = torch.pow(torch.clamp(net_image, 0, 1), 1.0/2.2)
                    net_image_bytes = memoryview((vis * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()
        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        
        
        render_pkg = original_render(viewpoint_cam, gaussians, pipe, background, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image = render_pkg["render"]
        gt_image = viewpoint_cam.original_image.cuda()

       
        patch_size = 512 
        _, H, W = image.shape
        if H > patch_size and W > patch_size:
            top = randint(0, H - patch_size)
            left = randint(0, W - patch_size)
            image_patch = image[:, top:top+patch_size, left:left+patch_size]
            gt_patch = gt_image[:, top:top+patch_size, left:left+patch_size]
        else:
            image_patch = image
            gt_patch = gt_image

       
        pred_gamma = torch.pow(image_patch + 1e-6, 1.0/2.2)
        gt_gamma = torch.pow(gt_patch + 1e-6, 1.0/2.2)

        
        Ll1 = l1_loss(pred_gamma, gt_gamma)
        
        L_edge = weighted_edge_loss(pred_gamma, gt_gamma)
        
        ssim_value = ssim(pred_gamma.unsqueeze(0), gt_gamma.unsqueeze(0))
        
        loss = (1.0 - opt.lambda_dssim) * Ll1 + \
               opt.lambda_dssim * (1.0 - ssim_value) + \
               0.1 * L_edge
        
        loss.backward()

    
        torch.nn.utils.clip_grad_norm_(gaussians._xyz, 1.0)

        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if (iteration in saving_iterations):
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[render_pkg["visibility_filter"]] = torch.max(gaussians.max_radii2D[render_pkg["visibility_filter"]], render_pkg["radii"][render_pkg["visibility_filter"]])
                gaussians.add_densification_stats(render_pkg["viewspace_points"], render_pkg["visibility_filter"])

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, render_pkg["radii"])
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

def prepare_output_and_logger(args):    
    if not args.model_path:
        args.model_path = os.path.join("./output/", str(uuid.uuid4()))
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    return None

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)
    safe_state(args.quiet)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)
    print("\nTraining complete.")