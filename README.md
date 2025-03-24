
1. Clone our repository

```
git clone https://github.com/CeviKle/KLETech-CEVI_LowLightEnhancement.git
cd KLETech-CEVI_LowLightEnhancement
```

2. Make conda environment

```
conda create -n sysuenv python=3.8
conda activate sysuenv
```

3. Install dependencies

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu111
pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm
pip install einops gdown addict future lmdb numpy pyyaml requests scipy tb-nightly yapf lpips
```

4. Install basicsr

```
python setup.py develop --no_cuda_ext
```

### Train

Set the dataset root of the configuration in `./Options/Ntire24UHDLowLight.yml` and then run

```
python basicsr/train.py
```

### Test

For testing your own data, you can run

```
cd Enhancement
python test.py --weights [your pretrained model weights] --input_dir [your input data path] --result_dir [your result saved path] --dataset [your dataset name]
```

We have placed our pre-trained model for this challenge in `Enhancement/pretrained_models/net_g_150000.pth`. **If you just want to run the challenge official input data, you can run** 

Download the pretrained weights from this [google drive](https://drive.google.com/drive/folders/1pjq-O2P7OgMBmJ_5o3eauU34OTMzVG-0?usp=sharing)
```
cd Enhancement
python test.py --input_dir [your input data path]
```


## Acknowledgments

This repo is built based on

- [**UHDM**](https://github.com/CVMI-Lab/UHDM) 
- **[MIRNet-v2](https://github.com/swz30/MIRNetv2/)**

We really appreciate their excellent works!

We also thank the computational sources supported by [Frontier Vision Lab](https://fvl2020.github.io/fvl.github.com/), SUN YAT-SEN University.
