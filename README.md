# NTIRE 2026 – Low-Light Image Enhancement Challenge

## Team: KLETech-CEVI

This repository presents our solution for the **NTIRE 2026 Low-Light Image Enhancement Challenge**.

Our approach is based on the **UHDM (Universal Hierarchical Decomposition Model)**, implemented using the **BasicSR framework**. The repository provides complete resources including training pipeline, inference scripts, pretrained models, and reproducibility instructions corresponding to our Codabench submission.

---

## 1. Overview

Low-light image enhancement is a critical task in computer vision, aiming to improve visibility and perceptual quality under challenging illumination conditions.

Our proposed method leverages a hierarchical decomposition strategy to effectively enhance low-light images while preserving structural details and minimizing artifacts.

---

## 2. Repository Structure

```
NTIRE2026-KLETech-CEVI-LLIE
│
├── Enhancement/                # Inference and evaluation pipeline
├── basicsr/                    # BasicSR framework
├── basicsr.egg-info/           # Build metadata
├── checkpoint/                 # Model checkpoints
├── setup.py                    # Installation script
├── test_1.py                   # Testing script
├── LICENSE                     # License file
├── VERSION                     # Version info
└── README.md                   # Documentation
```

---

## 3. Installation

### 3.1 Clone Repository

```bash
git clone https://github.com/CeviKle/NTIRE2026-KLETech-CEVI-LLIE.git
cd NTIRE2026-KLETech-CEVI-LLIE
```

### 3.2 Environment Setup

```bash
conda create -n sysuenv python=3.8
conda activate sysuenv
```

### 3.3 Install Dependencies

**PyTorch (CUDA 11.1):**
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu111
```

**Required Libraries:**
```bash
pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm
pip install einops gdown addict future lmdb numpy pyyaml requests scipy tb-nightly yapf lpips
```

### 3.4 Install BasicSR

```bash
python setup.py develop --no_cuda_ext
```

---

## 4. Dataset

The model is trained on the **official NTIRE 2026 Low-Light Image Enhancement dataset**.

Update dataset paths in:
```bash
Options/Ntire24UHDLowLight.yml
```

---

## 5. Training

```bash
python basicsr/train.py
```

---

## 6. Inference

### 6.1 General Inference

```bash
cd Enhancement
python test.py --weights [model_weights] --input_dir [input_images] --result_dir [output_folder]
```

### 6.2 Example

```bash
python test.py \
--weights pretrained_models/averaged_model.pth \
--input_dir ./input \
--result_dir ./results
```

---

## 7. Pretrained Models and Results

Download pretrained weights and results from:

https://drive.google.com/drive/folders/1HhO0yl2K4tHHgzxGv9mZRMYjqMzlGQ70?usp=sharing

### Contents:
- Final trained model weights  
- Results submitted to Codabench  
- Additional evaluation outputs  

Place the downloaded weights in:

```bash
Enhancement/pretrained_models/
```

---

## 8. Reproducibility (NTIRE Submission)

```bash
cd Enhancement

python test.py \
--weights pretrained_models/averaged_model.pth \
--input_dir ./data/input \
--result_dir ./results
```

Steps:
1. Download pretrained weights  
2. Place them in `Enhancement/pretrained_models/`  
3. Provide input images in `./data/input/`  
4. Run the command above  

The generated outputs will match our Codabench submission.

---

## 9. Method Description

Our solution is based on **UHDM**, designed for high-quality low-light enhancement.

### Key Components:
- Hierarchical decomposition-based architecture  
- BasicSR training pipeline  
- Checkpoint averaging  
- X8 self-ensemble during inference  
- Resolution-preserving enhancement  

---

## 10. Hardware Requirements

- GPU: NVIDIA GPU (recommended)  
- CUDA: 11.1  
- Framework: PyTorch  

---

## 11. Acknowledgements

This work builds upon the following contributions:

- UHDM  
- MIRNet-v2  
- BasicSR  

We thank the respective authors for their valuable work.

We also acknowledge computational support from:

Frontier Vision Lab, Sun Yat-Sen University

---

## 12. NTIRE 2026 Submission

This repository contains all necessary components to reproduce our NTIRE 2026 submission, including code, pretrained models, and evaluation pipeline.

---

## 13. Contact

KLE Technological University  

For any queries regarding this repository or the NTIRE 2026 submission, please contact the team.
