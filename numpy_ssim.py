import numpy as np
import cv2
import os
import glob

def calculate_ssim_numpy(img1, img2):
   
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    
    
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean()

def find_first_image(path):
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']
        for ext in extensions:
            files = glob.glob(os.path.join(path, ext))
            if files:
                return files[0]
    return None


render_input = r"D:\gaussian-splatting\output\alley\Fusion_GS\1_Standard_Gamma_2.2"
gt_input = r"D:\gaussian-splatting\alley\input"

render_path = find_first_image(render_input)
gt_path = find_first_image(gt_input)

if not render_path or not gt_path:
    print("❌ 错误：找不到图片文件！")
else:
    img_r = cv2.imread(render_path, cv2.IMREAD_GRAYSCALE)
    img_g = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

    if img_r is None or img_g is None:
        print("❌ 读取失败")
    else:
        if img_r.shape != img_g.shape:
            img_r = cv2.resize(img_r, (img_g.shape[1], img_g.shape[0]))

        score = calculate_ssim_numpy(img_r, img_g)
        
        print("-" * 30)
        print(f"🖼️  正在对比: {os.path.basename(render_path)}")
        print(f"✅ [修正后 NumPy 结果] SSIM: {score:.6f}")
        print("-" * 30)