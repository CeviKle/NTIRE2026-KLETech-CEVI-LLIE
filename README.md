[# NTIRE 2026 – Low-Light Image Enhancement Challenge]

### Team: KLETech-CEVI

This repository contains the implementation of our solution for the **NTIRE 2026 Low-Light Image Enhancement Challenge**.

Our method is based on the **UHDM architecture implemented using the BasicSR framework**.
The repository includes training code, inference scripts, and instructions to reproduce the results submitted to Codabench.

---

# Repository Structure

```
KLETech-CEVI_LowLightEnhancement
│
├── Enhancement/                # Inference scripts and evaluation pipeline
│
├── basicsr/                    # BasicSR framework implementation
│
├── basicsr.egg-info/           # BasicSR build information
│
├── checkpoint/                 # Training checkpoints and models
│
├── setup.py                    # Install BasicSR package
│
├── test_1.py                   # Testing / inference script
│
├── LICENSE                     # License file
│
├── VERSION                     # Version information
│
└── README.md                   # Project documentation
```

---

# Clone Repository

```
git clone https://github.com/CeviKle/KLETech-CEVI_LowLightEnhancement.git
cd KLETech-CEVI_LowLightEnhancement
```

---

# Environment Setup

Create a conda environment.

```
conda create -n sysuenv python=3.8
conda activate sysuenv
```

---

# Install Dependencies

Install PyTorch.

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu111
```

Install required libraries.

```
pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm
pip install einops gdown addict future lmdb numpy pyyaml requests scipy tb-nightly yapf lpips
```

---

# Install BasicSR

```
python setup.py develop --no_cuda_ext
```

---

# Training

Set the dataset root directory in:

```
Options/Ntire24UHDLowLight.yml
```

Then run training:

```
python basicsr/train.py
```

---

# Testing / Inference

For testing your own images:

```
cd Enhancement
python test.py --weights [model_weights] --input_dir [input_images] --result_dir [output_folder] --dataset [dataset_name]
```

Example:

```
python test.py \
--weights pretrained_models/averaged_model.pth \
--input_dir ./input \
--result_dir ./results
```

---

# Pretrained Model

The final trained model weights are available here:

**Google Drive (Weights + Results)**

https://drive.google.com/drive/folders/1HhO0yl2K4tHHgzxGv9mZRMYjqMzlGQ70?usp=sharing

The folder contains:

* Final model weights
* Inference results used for NTIRE submission
* Additional datasets used for testing

Download the weights and place them inside:

```
Enhancement/pretrained_models/
```

---

# Running Official Challenge Data

After downloading the pretrained model:

```
cd Enhancement
python test.py --input_dir [path_to_official_input]
```

Enhanced images will be saved in the `results` directory.

---

# Method Overview

Our approach uses **UHDM (Ultra High Definition Model)** for low-light enhancement.

Key components:

* BasicSR training framework
* UHDM architecture
* checkpoint averaging
* X8 self-ensemble inference
* resolution preserving output

The model is trained on the **NTIRE Low-Light Image Enhancement dataset**.

---

# Acknowledgements

This repository is built based on the following excellent works:

* UHDM
* MIRNet-v2
* BasicSR framework

We sincerely thank the authors for their contributions.

We also acknowledge the computational resources provided by:

Frontier Vision Lab
Sun Yat-Sen University

---

# NTIRE 2026 Submission

This repository provides the code and models required to reproduce our NTIRE 2026 submission.

The results submitted to Codabench can be reproduced using the provided pretrained weights and inference scripts.

---

# Contact

KLE Technological University

For questions regarding this repository or the NTIRE 2026 submission, please contact the team members.

](https://github.com/CeviKle/NTIRE2026-KLETech-CEVI-LLIE/blob/main/README.md?plain=1)
