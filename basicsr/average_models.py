import torch
import os

model_dir = r"\\10.9.0.114\cvg-archive\NTIRE2026\runs\C9_LLIE\KLETech-CEVI_LowLightEnhancement\forced_checkpoints_LLIE"

models = [
"net_g_241000.pth",
"net_g_242000.pth",
"net_g_243000.pth",
"net_g_244000.pth",
"net_g_246000.pth",
"net_g_248000.pth",
"net_g_249000.pth",
"net_g_250000.pth"
]

avg = None

for m in models:

    path = os.path.join(model_dir, m)
    print("Loading:", path)

    state = torch.load(path, map_location="cpu")

    if avg is None:
        avg = state
        for k in avg:
            avg[k] = avg[k].float()
    else:
        for k in avg:
            avg[k] += state[k].float()

for k in avg:
    avg[k] /= len(models)

save_path = os.path.join(model_dir, "averaged_model.pth")

torch.save(avg, save_path)

print("Saved averaged model:", save_path)