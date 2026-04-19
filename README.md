# LumeGS: Robust Low-Light 3D Gaussian Splatting via Logarithmic Structural Optimization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 
This is the official PyTorch implementation of **LumeGS** (*Neurocomputing 2026 Submission*). 

LumeGS is a purely mathematical optimization framework that resolves the geometric collapse of 3D Gaussian Splatting (3DGS) in extreme low-light (LLL) environments. It achieves state-of-the-art structural fidelity **without introducing any neural network overhead**, preserving the native **60+ FPS real-time rendering speed** on consumer-grade hardware (e.g., RTX 2060).

![LumeGS Pipeline](assets/pipeline.png) 
## 🚀 Key Features
- **Logarithmic Radiance Optimization ($\mathcal{L}_{\log}$)**: Adaptively magnifies gradient flow in dark regions ($|\nabla \mathcal{L}_{log}| \propto \frac{1}{\hat{I} + \epsilon}$), forcing accurate geometry densification where signals are weakest.
- **Gamma-Space Edge Regularization ($\mathcal{L}_{\text{edge}}$)**: Utilizes spatial high-pass constraints (Sobel filters) in a perceptually uniform space to anchor physical boundaries and aggressively suppress incoherent sensor noise floaters.
- **Zero-Cost Inference**: All illumination and structural enhancements are baked directly into the explicit 3D Gaussian primitives (Spherical Harmonics and opacity). **Absolutely zero extra FLOPs during rendering.**

## 🛠️ Installation

```bash
# 1. Clone the repository
git clone (https://github.com/)/LumeGS.git
cd LumeGS

# 2. Create a conda environment
conda create -n lumegs python=3.8
conda activate lumegs

# 3. Install PyTorch (Please adapt the CUDA version to your hardware)
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url [https://download.pytorch.org/whl/cu113](https://download.pytorch.org/whl/cu113)

# 4. Install additional requirements
pip install -r requirements.txt

# 5. Install the required submodules (Differentiable Rasterization)
# Note: You need to clone the official diff-gaussian-rasterization and simple-knn 
# from the original 3DGS repository and install them locally.
# pip install ./submodules/diff-gaussian-rasterization
# pip install ./submodules/simple-knn
