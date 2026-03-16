import cv2
import numpy as np
import torch
from basicsr.metrics.metric_util import reorder_image, to_y_channel


def calculate_psnr(img1,
                   img2,
                   crop_border,
                   input_order='HWC',
                   test_y_channel=False):

    import numpy as np
    import torch

    # -------- Convert Tensor to numpy safely -------- #
    def tensor_to_numpy(img):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()

        # remove batch dim only if present
        if img.ndim == 4 and img.shape[0] == 1:
            img = img[0]

        # if CHW → convert to HWC
        if img.ndim == 3 and img.shape[0] in [1, 3]:
            img = np.transpose(img, (1, 2, 0))

        return img

    img1 = tensor_to_numpy(img1)
    img2 = tensor_to_numpy(img2)

    assert img1.shape == img2.shape, (
        f'Shape mismatch: {img1.shape}, {img2.shape}')

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')

    max_value = 1.0 if img1.max() <= 1 else 255.0
    return 20. * np.log10(max_value / np.sqrt(mse))



def calculate_ssim(img1,
                   img2,
                   crop_border,
                   input_order='HWC',
                   test_y_channel=False):

    import numpy as np
    import torch
    from skimage.metrics import structural_similarity as ssim

    def tensor_to_numpy(img):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()

        if img.ndim == 4 and img.shape[0] == 1:
            img = img[0]

        if img.ndim == 3 and img.shape[0] in [1, 3]:
            img = np.transpose(img, (1, 2, 0))

        return img

    img1 = tensor_to_numpy(img1)
    img2 = tensor_to_numpy(img2)

    assert img1.shape == img2.shape, (
        f'Shape mismatch: {img1.shape}, {img2.shape}')

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    max_value = 1.0 if img1.max() <= 1 else 255.0

    return ssim(img1, img2, data_range=max_value, channel_axis=-1)
