import torch
import os

model_dir = "/NTIRE2026/runs/C9_LLIE/KLETech-CEVI_LowLightEnhancement/forced_checkpoints_LLIE"

models = [
"net_g_244000.pth",
"net_g_246000.pth",
"net_g_248000.pth",
"net_g_249000.pth",
"net_g_250000.pth"
]

avg_state = None

for m in models:

    path = os.path.join(model_dir, m)
    print("Loading:", path)

    ckpt = torch.load(path, map_location="cpu")

    if "params" in ckpt:
        state = ckpt["params"]
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    if avg_state is None:
        avg_state = {k: v.float() for k, v in state.items()}
    else:
        for k in avg_state:
            avg_state[k] += state[k].float()

for k in avg_state:
    avg_state[k] /= len(models)

save_path = os.path.join(model_dir, "averaged_model.pth")

torch.save({"params": avg_state}, save_path)

print("Saved averaged model:", save_path)