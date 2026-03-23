"""Shared diffusion schedules and loss helpers for training/inference."""

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def make_cosine_schedule(T: int = 1000):
    s = 0.008
    t = torch.linspace(0, T, T + 1)
    f = torch.cos(((t / T) + s) / (1 + s) * np.pi / 2) ** 2
    abar = f / f[0]
    betas = torch.clamp(1 - (abar[1:] / abar[:-1]), 1e-5, 0.999)
    alphas = 1.0 - betas
    abar = torch.cumprod(alphas, dim=0)
    return betas, alphas, abar


@torch.no_grad()
def compute_latent_stats(encoder, dataloader, device):
    sum_means = None
    sum_stds = None
    n_batches = 0

    for batch in tqdm(dataloader):
        imgs, gt_depth = batch[0], batch[1]
        imgs = imgs.to(device)
        gt_depth = gt_depth.to(device)

        mu, _ = encoder(gt_depth)

        mu_flat = mu.permute(1, 0, 2, 3).contiguous().view(mu.shape[1], -1)
        mean_c = mu_flat.mean(dim=1)
        std_c = mu_flat.std(dim=1)

        if sum_means is None:
            sum_means = mean_c
            sum_stds = std_c
        else:
            sum_means += mean_c
            sum_stds += std_c

        n_batches += 1

    mean_final = sum_means / n_batches
    std_final = sum_stds / n_batches
    return mean_final, std_final, mu


def gradient_loss(pred, gt, mask: torch.Tensor | None = None):
    if mask is None:
        mask = gt > 1e-6
    else:
        mask = mask > 0.5

    epsilon = 1e-8

    gt = gt + 1e-8
    dx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dx_gt = gt[:, :, :, 1:] - gt[:, :, :, :-1]
    dy_gt = gt[:, :, 1:, :] - gt[:, :, :-1, :]

    mask_x = (mask[:, :, :, 1:] & mask[:, :, :, :-1]).float()
    mask_y = (mask[:, :, 1:, :] & mask[:, :, :-1, :]).float()

    loss_dx_per_pixel = torch.abs(dx_pred - dx_gt) * mask_x
    sum_loss_dx = loss_dx_per_pixel.sum(dim=[1, 2, 3])
    count_dx = mask_x.sum(dim=[1, 2, 3])
    per_sample_loss_dx = sum_loss_dx / (count_dx + epsilon)

    loss_dy_per_pixel = torch.abs(dy_pred - dy_gt) * mask_y
    sum_loss_dy = loss_dy_per_pixel.sum(dim=[1, 2, 3])
    count_dy = mask_y.sum(dim=[1, 2, 3])
    per_sample_loss_dy = sum_loss_dy / (count_dy + epsilon)

    return per_sample_loss_dx + per_sample_loss_dy


def absolute_loss(pred, gt):
    mask = gt > 1e-6

    gt = gt + 1e-8
    per_pixel_loss = F.l1_loss(pred, gt, reduction="none")
    masked_per_pixel_loss = per_pixel_loss * mask.float()

    sum_of_losses = masked_per_pixel_loss.sum(dim=[1, 2, 3])
    num_pixels = mask.sum(dim=[1, 2, 3])

    per_sample_loss = sum_of_losses / (num_pixels + 1e-8)

    return per_sample_loss


def masked_tv_score(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Lower is smoother/more coherent inside mask. Returns per-sample scores."""
    mask_b = mask > 0.5
    dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]

    mask_x = (mask_b[:, :, :, 1:] & mask_b[:, :, :, :-1]).float()
    mask_y = (mask_b[:, :, 1:, :] & mask_b[:, :, :-1, :]).float()

    tv_x = (dx.abs() * mask_x).sum(dim=[1, 2, 3]) / (mask_x.sum(dim=[1, 2, 3]) + 1e-8)
    tv_y = (dy.abs() * mask_y).sum(dim=[1, 2, 3]) / (mask_y.sum(dim=[1, 2, 3]) + 1e-8)
    return tv_x + tv_y
