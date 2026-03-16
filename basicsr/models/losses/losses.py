import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from collections import OrderedDict

# ------------------------------------------------------------
# Basic Losses
# ------------------------------------------------------------

class L1Loss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target):
        loss = F.l1_loss(pred, target, reduction=self.reduction)
        return self.loss_weight * loss


class MSELoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target):
        loss = F.mse_loss(pred, target, reduction=self.reduction)
        return self.loss_weight * loss


# ------------------------------------------------------------
# PSNR Loss (Corrected)
# ------------------------------------------------------------

class PSNRLoss(nn.Module):
    """
    Minimizing this is equivalent to maximizing PSNR.
    """

    def __init__(self, loss_weight=1.0, toY=False):
        super().__init__()
        self.loss_weight = loss_weight
        self.toY = toY

        # RGB -> Y conversion coefficients
        self.register_buffer(
            "coef",
            torch.tensor([65.481, 128.553, 24.966]).view(1, 3, 1, 1)
        )

    def forward(self, pred, target):
        assert pred.ndim == 4

        if self.toY:
            pred = (pred * self.coef).sum(dim=1, keepdim=True) + 16.
            target = (target * self.coef).sum(dim=1, keepdim=True) + 16.
            pred = pred / 255.
            target = target / 255.

        mse = ((pred - target) ** 2).mean(dim=(1, 2, 3))
        psnr = -10.0 * torch.log10(mse + 1e-8)

        return self.loss_weight * psnr.mean()


# ------------------------------------------------------------
# Charbonnier Loss
# ------------------------------------------------------------

class CharbonnierLoss(nn.Module):
    """Differentiable L1"""

    def __init__(self, loss_weight=1.0, eps=1e-3):
        super().__init__()
        self.loss_weight = loss_weight
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.mean(torch.sqrt(diff * diff + self.eps ** 2))
        return self.loss_weight * loss


# ------------------------------------------------------------
# VGG Perceptual Loss (Optimized)
# ------------------------------------------------------------

class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super().__init__()

        vgg = torchvision.models.vgg16(
            weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1
        ).features.eval()

        self.blocks = nn.ModuleList([
            vgg[:4],
            vgg[4:9],
            vgg[9:16],
            vgg[16:23]
        ])

        for block in self.blocks:
            for param in block.parameters():
                param.requires_grad = False

        self.resize = resize

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, input, target, feature_layers=[0,1,2,3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std

        if self.resize:
            input = F.interpolate(input, size=(224, 224), mode='bilinear', align_corners=False)
            target = F.interpolate(target, size=(224, 224), mode='bilinear', align_corners=False)

        loss = 0.0
        x = input
        y = target

        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)

            if i in feature_layers:
                loss += F.l1_loss(x, y)

            if i in style_layers:
                act_x = x.view(x.size(0), x.size(1), -1)
                act_y = y.view(y.size(0), y.size(1), -1)

                gram_x = act_x @ act_x.transpose(1, 2)
                gram_y = act_y @ act_y.transpose(1, 2)

                loss += F.l1_loss(gram_x, gram_y)

        return loss


# ------------------------------------------------------------
# Multi-scale Perceptual Loss
# ------------------------------------------------------------

class MultiVGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.per_loss = VGGPerceptualLoss()
        self.charb = CharbonnierLoss()

    def forward(self, out1, out2, out3, gt):
        gt2 = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=False)
        gt3 = F.interpolate(gt, scale_factor=0.25, mode='bilinear', align_corners=False)

        l1_1 = self.charb(out1, gt)
        l1_2 = self.charb(out2, gt2)
        l1_3 = self.charb(out3, gt3)

        p_1 = 0.04 * self.per_loss(out1, gt)
        p_2 = 0.04 * self.per_loss(out2, gt2)
        p_3 = 0.04 * self.per_loss(out3, gt3)

        total = (l1_1 + p_1) + (l1_2 + p_2) + (l1_3 + p_3)

        loss_dict = OrderedDict({
            "charbonnier": l1_1 + l1_2 + l1_3,
            "perceptual": p_1 + p_2 + p_3,
            "total": total
        })

        return total, loss_dict


# ------------------------------------------------------------
# FFT Loss (Fixed)
# ------------------------------------------------------------

class FFTLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target):
        pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
        target_fft = torch.fft.fft2(target, dim=(-2, -1))

        pred_fft = torch.stack([pred_fft.real, pred_fft.imag], dim=-1)
        target_fft = torch.stack([target_fft.real, target_fft.imag], dim=-1)

        loss = F.l1_loss(pred_fft, target_fft, reduction=self.reduction)
        return self.loss_weight * loss


# ------------------------------------------------------------
# Total Variation Loss
# ------------------------------------------------------------
class TVLoss(torch.nn.Module):

    def __init__(self, loss_weight=1.0):
        super(TVLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, x):

        # If network outputs multi-stage predictions
        if isinstance(x, (list, tuple)):
            x = x[-1]   # take final prediction

        # Total Variation
        h_tv = torch.mean((x[:, :, 1:, :] - x[:, :, :-1, :]) ** 2)
        w_tv = torch.mean((x[:, :, :, 1:] - x[:, :, :, :-1]) ** 2)

        loss = h_tv + w_tv

        return loss * float(self.loss_weight)