import torch
import pyiqa
import os
import argparse
from tqdm import tqdm
import glob
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_dir", type=str, required=True, help="Path to your rendered images (LumeGS)")
parser.add_argument("-d", "--device", type=str, default="cuda", help="Device to use: cuda or cpu")
args = parser.parse_args()


device = torch.device(args.device if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")


print("⏳ Loading NIQE model...")

niqe_metric = pyiqa.create_metric('niqe', device=device)

print("⏳ Loading MUSIQ model...")

musiq_metric = pyiqa.create_metric('musiq', device=device, as_loss=False)


img_paths = sorted(glob.glob(os.path.join(args.input_dir, "*.[jJ][pP][gG]")) + 
                   glob.glob(os.path.join(args.input_dir, "*.[pP][nN][gG]")) +
                   glob.glob(os.path.join(args.input_dir, "*.[jJ][pP][eE][gG]")))

if len(img_paths) == 0:
    print("❌ Error: No images found in the input directory!")
    exit()

niqe_scores = []
musiq_scores = []

print(f"✅ Found {len(img_paths)} images. Starting evaluation...")

for img_path in tqdm(img_paths):
    try:
    
        n_score = niqe_metric(img_path).item()
        
        m_score = musiq_metric(img_path).item()
        
        niqe_scores.append(n_score)
        musiq_scores.append(m_score)
        
    except Exception as e:
        print(f"⚠️ Warning: Failed to process {img_path}. Error: {e}")


avg_niqe = np.mean(niqe_scores)
avg_musiq = np.mean(musiq_scores)

print("\n" + "="*40)
print(" 📊 FINAL NO-REFERENCE METRICS")
print("="*40)
print(f" Image Count : {len(niqe_scores)}")
print("-" * 40)
print(f" NIQE  (↓ Lower is better)  : {avg_niqe:.4f}")
print(f" MUSIQ (↑ Higher is better) : {avg_musiq:.4f}")
print("="*40)

if avg_niqe < 4.0:
    print("✨ Great! NIQE < 4.0 usually implies good naturalness.")
else:
    print("👉 Note: NIQE > 5.0 might indicate noise or unnatural artifacts.")