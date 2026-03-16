import numpy as np
import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from natsort import natsorted
from glob import glob
from basicsr.models.archs.UHDM_arch import UHDM
from skimage import img_as_ubyte
from PIL import Image
import yaml

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, required=True)
parser.add_argument('--result_dir', type=str, required=True)
parser.add_argument('--weights', type=str, required=True)
args = parser.parse_args()

# -------- LOAD YAML --------
yaml_file = 'Options/Ntire24UHDLowLight.yml'

with open(yaml_file, 'r') as f:
    opt = yaml.safe_load(f)

opt['network_g'].pop('type', None)
opt['network_g'].pop('path', None)

model = UHDM(**opt['network_g'])

# -------- LOAD CHECKPOINT --------
checkpoint = torch.load(args.weights, map_location='cpu')

if 'params' in checkpoint:
    model.load_state_dict(checkpoint['params'])
elif 'state_dict' in checkpoint:
    model.load_state_dict(checkpoint['state_dict'])
else:
    model.load_state_dict(checkpoint)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

model.eval()

# --------------------------------
# X8 SELF ENSEMBLE FUNCTION
# --------------------------------

def transform(v, op):
    if op == 'v':
        v = torch.flip(v, [2])
    elif op == 'h':
        v = torch.flip(v, [3])
    elif op == 't':
        v = v.transpose(2, 3)
    return v


def forward_x8(model, x):

    lr_list = [x]

    for tf in ['v', 'h', 't']:
        lr_list.extend([transform(t, tf) for t in lr_list])

    sr_list = []

    for aug in lr_list:

        out = model(aug)

        if isinstance(out, (list, tuple)):
            out = out[0]

        if isinstance(out, dict):
            out = out['out'] if 'out' in out else list(out.values())[0]

        sr_list.append(out)

    for i in range(len(sr_list)):

        if i > 3:
            sr_list[i] = transform(sr_list[i], 't')

        if i % 4 > 1:
            sr_list[i] = transform(sr_list[i], 'h')

        if (i % 4) % 2 == 1:
            sr_list[i] = transform(sr_list[i], 'v')

    output = torch.stack(sr_list, dim=0).mean(dim=0)

    return output


# -------- LOAD INPUT IMAGES --------

input_paths = natsorted(
    glob(os.path.join(args.input_dir, '*.jpg')) +
    glob(os.path.join(args.input_dir, '*.png'))
)

os.makedirs(args.result_dir, exist_ok=True)

# -------- INFERENCE --------

with torch.inference_mode():

    for inp_path in tqdm(input_paths):

        # LOAD ORIGINAL IMAGE
        orig_pil = Image.open(inp_path).convert("RGB")
        orig_w, orig_h = orig_pil.size

        img = np.float32(np.array(orig_pil)) / 255.

        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

        # -------- MODEL WITH X8 ENSEMBLE --------

        restored = forward_x8(model, img_tensor)

        if isinstance(restored, dict):
            restored = restored['out'] if 'out' in restored else list(restored.values())[0]

        if isinstance(restored, (list, tuple)):
            restored = restored[0]

        restored = restored.squeeze(0)
        restored = torch.clamp(restored, 0, 1).cpu().permute(1, 2, 0).numpy()

        restored_uint8 = img_as_ubyte(restored)

        # -------- FORCE FINAL RESIZE --------

        restored_pil = Image.fromarray(restored_uint8)
        restored_pil = restored_pil.resize((orig_w, orig_h), Image.BICUBIC)

        filename = os.path.splitext(os.path.basename(inp_path))[0] + ".png"
        save_path = os.path.join(args.result_dir, filename)

        restored_pil.save(save_path)

print("Inference completed successfully.")