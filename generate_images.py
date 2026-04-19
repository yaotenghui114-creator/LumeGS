import os
import torch
from gaussian_renderer import render as original_render 
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import torchvision

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def apply_gamma(image, gamma_val):
  

    image = torch.clamp(image, 0.0, 1.0)
    

    return torch.pow(image + 1e-6, gamma_val)

def generate(dataset, opt, pipe, checkpoint, iteration):
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians, load_iteration=-1, shuffle=False)
    
    if not hasattr(gaussians, "_exposure"):
        gaussians._exposure = torch.zeros((1, 3, 3), device="cuda", requires_grad=True)
    gaussians.training_setup(opt)
    
    if os.path.exists(checkpoint):
        print(f"Loading Checkpoint: {checkpoint}")
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    else:
        # Fallback to PLY
        ply_path = os.path.join(dataset.model_path, "point_cloud", "iteration_{}".format(iteration), "point_cloud.ply")
        if os.path.exists(ply_path):
            print(f"Loading PLY: {ply_path}")
            gaussians.load_ply(ply_path)
        else:
            print("Error: Model not found.")
            return

    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

  
    path_std = os.path.join(dataset.model_path, "1_Standard_Gamma_2.2")
    path_soft = os.path.join(dataset.model_path, "2_Soft_Gamma_1.8")
    path_bright = os.path.join(dataset.model_path, "3_Bright_Gamma_1.5")
    
    os.makedirs(path_std, exist_ok=True)
    os.makedirs(path_soft, exist_ok=True)
    os.makedirs(path_bright, exist_ok=True)

    cameras = scene.getTrainCameras() + scene.getTestCameras()

    print(f"Rendering {len(cameras)} images with Adaptive Gamma...")
    for idx, view in enumerate(tqdm(cameras, desc="Rendering")):
        with torch.no_grad():
            render_pkg = original_render(view, gaussians, pipe, background, 
                                       use_trained_exp=dataset.train_test_exp, 
                                       separate_sh=SPARSE_ADAM_AVAILABLE)
            
          
            image = render_pkg["render"]
            
        
            img_std = apply_gamma(image, 1.0/2.2)
            torchvision.utils.save_image(img_std, os.path.join(path_std, f"{view.image_name}.png"))
            
        
            img_soft = apply_gamma(image, 1.0/1.8)
            torchvision.utils.save_image(img_soft, os.path.join(path_soft, f"{view.image_name}.png"))
            
        
            img_bright = apply_gamma(image, 1.0/1.5)
            torchvision.utils.save_image(img_bright, os.path.join(path_bright, f"{view.image_name}.png"))

if __name__ == "__main__":
    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    
    print(f"Generating images for {args.model_path}...")
    ckpt_path = os.path.join(args.model_path, "chkpnt" + str(args.iteration) + ".pth")
    generate(lp.extract(args), op.extract(args), pp.extract(args), ckpt_path, args.iteration)