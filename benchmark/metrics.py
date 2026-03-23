from __future__ import annotations
import torch


NEAR = 3.0
FAR = 7.0


def norm_disp_to_depth(d: torch.Tensor, near: float = NEAR, far: float = FAR, eps: float = 1e-8) -> torch.Tensor:
    disp_near = 1.0 / near
    disp_far = 1.0 / far
    raw_disp = d * (disp_near - disp_far) + disp_far
    depth = 1.0 / (raw_disp + eps)
    return depth


def depth_to_norm_disp(depth: torch.Tensor, near: float = NEAR, far: float = FAR, eps: float = 1e-8) -> torch.Tensor:
    disp_near = 1.0 / near
    disp_far = 1.0 / far
    raw_disp = 1.0 / (depth + eps)
    norm_disp = (raw_disp - disp_far) / (disp_near - disp_far + eps)
    return torch.clamp(norm_disp, 0.0, 1.0)


def _disp_metrics_from_disp(
    pred_disp: torch.Tensor,
    gt_disp: torch.Tensor,
    mask: torch.Tensor | None = None,
    range_mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> dict:
    if mask is None:
        mask = gt_disp > eps
    if mask.sum() == 0:
        return {
            "mae": float("inf"),
            "mae_range_norm": float("inf"),
            "absrel": float("inf"),
            "d1": 0.0,
            "d2": 0.0,
            "d3": 0.0,
        }

    pd = pred_disp[mask]
    gd = gt_disp[mask]

    mae = (pd - gd).abs().mean().item()

    # Normalize by full-foreground GT disparity range for the item, independent
    # of the evaluation mask used for MAE.
    if range_mask is None:
        range_mask = gt_disp > eps
    range_valid = range_mask & torch.isfinite(gt_disp)
    if range_valid.sum() == 0:
        range_den = 1e-6
    else:
        range_den = (gt_disp[range_valid].max() - gt_disp[range_valid].min()).clamp_min(1e-6).item()
    mae_range_norm = mae / range_den

    absrel = ((pd - gd).abs() / (gd + 1e-8)).mean().item()

    ratio = torch.maximum(pd / (gd + 1e-8), gd / (pd + 1e-8))
    d1 = (ratio < 1.25).float().mean().item()
    d2 = (ratio < 1.25 ** 2).float().mean().item()
    d3 = (ratio < 1.25 ** 3).float().mean().item()

    return {"mae": mae, "mae_range_norm": mae_range_norm, "absrel": absrel, "d1": d1, "d2": d2, "d3": d3}


def disp_metrics(
    pred_norm_disp: torch.Tensor,
    gt_norm_disp: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> dict:
    """Non-symmetric metrics in normalized disparity space.

    This is useful for debugging partial-depth conditioning behavior.
    Prefer `disp_metrics_symmetric` for the main benchmark headline numbers.
    """
    fg_mask = gt_norm_disp > eps
    if mask is None:
        mask = fg_mask
    else:
        mask = mask & fg_mask
    return _disp_metrics_from_disp(pred_norm_disp, gt_norm_disp, mask=mask, range_mask=fg_mask, eps=eps)


def _depth_metrics_from_depth(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> dict:
    if mask is None:
        mask = (gt_depth > eps)
    if mask.sum() == 0:
        return {"mae": float("inf"), "absrel": float("inf"), "d1": 0.0, "d2": 0.0, "d3": 0.0}

    pd = pred_depth[mask]
    gd = gt_depth[mask]

    mae = (pd - gd).abs().mean().item()
    absrel = ((pd - gd).abs() / (gd + 1e-8)).mean().item()

    ratio = torch.maximum(pd / (gd + 1e-8), gd / (pd + 1e-8))
    d1 = (ratio < 1.25).float().mean().item()
    d2 = (ratio < 1.25 ** 2).float().mean().item()
    d3 = (ratio < 1.25 ** 3).float().mean().item()

    return {"mae": mae, "absrel": absrel, "d1": d1, "d2": d2, "d3": d3}


def depth_metrics(
    pred_norm_disp: torch.Tensor,
    gt_norm_disp: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> dict:
    """Non-symmetric metrics in depth space (derived from normalized disparity)."""
    pred_depth = norm_disp_to_depth(pred_norm_disp)
    gt_depth = norm_disp_to_depth(gt_norm_disp)

    fg_mask = gt_norm_disp > eps
    if mask is None:
        mask = fg_mask
    else:
        mask = mask & fg_mask
    return _depth_metrics_from_depth(pred_depth, gt_depth, mask=mask, eps=eps)


def depth_metrics_symmetric(
    pred_norm_disp: torch.Tensor,
    gt_norm_disp: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> dict:
    """
    Symmetric metrics over flips + depth inversion (like fully_symmetric_depth_mae).
    For MAE/AbsRel: take the minimum over transforms.
    For δ metrics: take the maximum over transforms.
    """
    pred_depth = norm_disp_to_depth(pred_norm_disp)
    gt_depth = norm_disp_to_depth(gt_norm_disp)

    # Use foreground mask from normalized disparity to avoid counting background
    # (norm_disp=0 maps to FAR depth, which would otherwise be included).
    fg_mask = gt_norm_disp > eps
    if mask is None:
        mask = fg_mask
    else:
        mask = mask & fg_mask

    if mask.sum() == 0:
        return {"mae": float("inf"), "absrel": float("inf"), "d1": 0.0, "d2": 0.0, "d3": 0.0}

    pred_mean = pred_depth[mask].mean()

    transforms = [
        ("original", pred_depth),
        ("depth_inv", 2 * pred_mean - pred_depth),
        ("hflip", torch.flip(pred_depth, dims=[-1])),
        ("hflip_depth_inv", torch.flip(2 * pred_mean - pred_depth, dims=[-1])),
        ("vflip", torch.flip(pred_depth, dims=[-2])),
        ("vflip_depth_inv", torch.flip(2 * pred_mean - pred_depth, dims=[-2])),
        ("hvflip", torch.flip(pred_depth, dims=[-2, -1])),
        ("hvflip_depth_inv", torch.flip(2 * pred_mean - pred_depth, dims=[-2, -1])),
    ]

    best = {"mae": float("inf"), "absrel": float("inf"), "d1": 0.0, "d2": 0.0, "d3": 0.0}
    for _, t_pred in transforms:
        m = _depth_metrics_from_depth(t_pred, gt_depth, mask=mask, eps=eps)
        best["mae"] = min(best["mae"], m["mae"])
        best["absrel"] = min(best["absrel"], m["absrel"])
        best["d1"] = max(best["d1"], m["d1"])
        best["d2"] = max(best["d2"], m["d2"])
        best["d3"] = max(best["d3"], m["d3"])

    return best


def disp_metrics_symmetric(
    pred_norm_disp: torch.Tensor,
    gt_norm_disp: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> dict:
    """
    Symmetric metrics computed directly in normalized disparity space.
    For MAE/AbsRel: take the minimum over transforms.
    For δ metrics: take the maximum over transforms.
    """
    fg_mask = gt_norm_disp > eps
    if mask is None:
        mask = fg_mask
    else:
        mask = mask & fg_mask

    if mask.sum() == 0:
        return {
            "mae": float("inf"),
            "mae_range_norm": float("inf"),
            "absrel": float("inf"),
            "d1": 0.0,
            "d2": 0.0,
            "d3": 0.0,
        }

    pred_mean = pred_norm_disp[mask].mean()

    transforms = [
        ("original", pred_norm_disp),
        ("disp_inv", 2 * pred_mean - pred_norm_disp),
        ("hflip", torch.flip(pred_norm_disp, dims=[-1])),
        ("hflip_disp_inv", torch.flip(2 * pred_mean - pred_norm_disp, dims=[-1])),
        ("vflip", torch.flip(pred_norm_disp, dims=[-2])),
        ("vflip_disp_inv", torch.flip(2 * pred_mean - pred_norm_disp, dims=[-2])),
        ("hvflip", torch.flip(pred_norm_disp, dims=[-2, -1])),
        ("hvflip_disp_inv", torch.flip(2 * pred_mean - pred_norm_disp, dims=[-2, -1])),
    ]

    best = {"mae": float("inf"), "mae_range_norm": float("inf"), "absrel": float("inf"), "d1": 0.0, "d2": 0.0, "d3": 0.0}
    for _, t_pred in transforms:
        m = _disp_metrics_from_disp(t_pred, gt_norm_disp, mask=mask, range_mask=fg_mask, eps=eps)
        best["mae"] = min(best["mae"], m["mae"])
        best["mae_range_norm"] = min(best["mae_range_norm"], m["mae_range_norm"])
        best["absrel"] = min(best["absrel"], m["absrel"])
        best["d1"] = max(best["d1"], m["d1"])
        best["d2"] = max(best["d2"], m["d2"])
        best["d3"] = max(best["d3"], m["d3"])

    return best
