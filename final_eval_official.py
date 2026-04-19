import torch
import lpips
import numpy as np
import cv2
import os
import glob
import argparse
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim_skimage
from skimage.metrics import peak_signal_noise_ratio as psnr_skimage



def prepare_image_for_lpips(img_bgr):
    """
    将 OpenCV 读取的 BGR 图片转换为 LPIPS 需要的 Tensor
    范围: [-1, 1], 格式: (1, 3, H, W)
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # 归一化到 [0, 1]
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    # 归一化到 [-1, 1]
    img_tensor = img_tensor * 2.0 - 1.0
    return img_tensor.unsqueeze(0)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--renders", type=str, required=True, help="你的渲染图文件夹 (LumeGS)")
    parser.add_argument("-g", "--gt", type=str, required=True, help="真值图文件夹 (推荐用 Gamma 提亮后的 input_gamma)")
    args = parser.parse_args()

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Initializing LPIPS on {device}...")
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    loss_fn_alex.eval()

   
    if not os.path.exists(args.renders) or not os.path.exists(args.gt):
        print("❌ 错误：文件夹路径不存在！")
        exit()

    r_files = sorted(glob.glob(os.path.join(args.renders, "*")))
    
    g_files_map = {os.path.splitext(os.path.basename(f))[0].split('.')[0]: f for f in glob.glob(os.path.join(args.gt, "*"))}

    psnr_list = []
    ssim_list = []
    lpips_list = []

    print(f"📂 正在扫描文件并计算指标...")
    
    count = 0
    for r_path in tqdm(r_files):
        if not os.path.isfile(r_path): continue
        
       
        stem = os.path.splitext(os.path.basename(r_path))[0].split('.')[0]
        
        if stem not in g_files_map:
            continue
            
        gt_path = g_files_map[stem]
        
       
        img_r = cv2.imread(r_path)
        img_g = cv2.imread(gt_path)
        
        if img_r is None or img_g is None: continue

        
        if img_r.shape != img_g.shape:
            img_r = cv2.resize(img_r, (img_g.shape[1], img_g.shape[0]))

       
        p_score = psnr_skimage(img_g, img_r, data_range=255)
        psnr_list.append(p_score)

        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        gray_g = cv2.cvtColor(img_g, cv2.COLOR_BGR2GRAY)
        
        s_score = ssim_skimage(gray_g, gray_r, data_range=255)
        ssim_list.append(s_score)

        with torch.no_grad():
            t_r = prepare_image_for_lpips(img_r).to(device)
            t_g = prepare_image_for_lpips(img_g).to(device)
            l_score = loss_fn_alex(t_r, t_g).item()
            lpips_list.append(l_score)
            
        count += 1

    if count == 0:
        print("❌ 未找到匹配的图片，请检查文件名是否对应！")
        exit()

    print("\n" + "="*50)
    print(f"📊 ACADEMIC STANDARD METRICS (skimage + lpips)")
    print(f"   Image Count : {count}")
    print("-" * 50)
    print(f"   PSNR  (↑) : {np.mean(psnr_list):.4f}")
    print(f"   SSIM  (↑) : {np.mean(ssim_list):.4f}")
    print(f"   LPIPS (↓) : {np.mean(lpips_list):.4f}")
    print("="*50)