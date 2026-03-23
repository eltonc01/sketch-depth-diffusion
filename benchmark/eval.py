from __future__ import annotations

import json
import os
import re
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from sketch_recon.models.vae_unet_control import VAE, DenoisingUNet2DConditionModel
from sketch_recon.training.train_diffusion import SketchDenoiserModule
from benchmark.metrics import disp_metrics_symmetric, norm_disp_to_depth, depth_to_norm_disp


def _infer_unet_config_from_checkpoint(model_checkpoint: str) -> dict:
    ckpt = torch.load(model_checkpoint, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    hparams = ckpt.get("hyper_parameters", {}) if isinstance(ckpt, dict) else {}

    unet_state = {k: v for k, v in state.items() if k.startswith("unet.")}
    if "unet.in_conv.weight" not in unet_state:
        raise KeyError("Checkpoint does not contain unet.in_conv.weight; cannot infer UNet config.")

    in_conv_w = unet_state["unet.in_conv.weight"]
    model_channels = int(in_conv_w.shape[0])

    inferred_use_controlnet = any(k.startswith("unet.zero_convs.") for k in unet_state.keys())
    use_controlnet = bool(hparams.get("use_controlnet", inferred_use_controlnet))

    in_channels = int(in_conv_w.shape[1])
    latent_channels = in_channels if use_controlnet else max(1, in_channels // 2)

    stage_resblock_indices: dict[int, list[int]] = {}
    stage_out_ch: dict[int, int] = {}
    for k in unet_state.keys():
        m = re.match(r"^unet\.downs\.(\d+)\.(\d+)\.time_proj\.weight$", k)
        if m is None:
            continue
        stage_idx = int(m.group(1))
        block_idx = int(m.group(2))
        stage_resblock_indices.setdefault(stage_idx, []).append(block_idx)
        conv1_key = f"unet.downs.{stage_idx}.{block_idx}.conv1.weight"
        if conv1_key in unet_state:
            stage_out_ch.setdefault(stage_idx, int(unet_state[conv1_key].shape[0]))

    if not stage_out_ch:
        raise ValueError("Failed to infer UNet down stages from checkpoint state dict.")

    stage_ids = sorted(stage_out_ch.keys())
    channel_mult = tuple(max(1, int(round(stage_out_ch[s] / max(model_channels, 1)))) for s in stage_ids)
    num_res_blocks = int(min(len(set(stage_resblock_indices.get(s, []))) for s in stage_ids))
    if num_res_blocks < 1:
        num_res_blocks = 1

    init_res = 32
    control_resolutions: list[int] = []
    for k in unet_state.keys():
        m = re.match(r"^unet\.zero_convs\.res_(\d+)\.conv\.weight$", k)
        if m is not None:
            control_resolutions.append(int(m.group(1)))
    if control_resolutions:
        init_res = int(max(control_resolutions))
    elif "img_size" in hparams:
        try:
            init_res = int(round(float(hparams["img_size"]) / 8.0))
        except Exception:
            init_res = 32

    attn_resolutions_set: set[int] = set()
    for k in unet_state.keys():
        m = re.match(r"^unet\.downs\.(\d+)\.(\d+)\.qkv\.weight$", k)
        if m is None:
            continue
        stage_idx = int(m.group(1))
        res = max(1, int(round(init_res / (2 ** stage_idx))))
        attn_resolutions_set.add(res)

    if "unet.mid_attn.qkv.weight" in unet_state:
        bottleneck_stage = max(len(channel_mult) - 1, 0)
        bottleneck_res = max(1, int(round(init_res / (2 ** bottleneck_stage))))
        attn_resolutions_set.add(bottleneck_res)

    if not attn_resolutions_set:
        attn_resolutions_set = {8, 16}
    attn_resolutions = tuple(sorted(attn_resolutions_set))

    control_channels = None
    if use_controlnet:
        inferred_control_channels: dict[int, int] = {}
        for k, v in unet_state.items():
            m = re.match(r"^unet\.zero_convs\.res_(\d+)\.conv\.weight$", k)
            if m is None:
                continue
            inferred_control_channels[int(m.group(1))] = int(v.shape[1])
        control_channels = inferred_control_channels if inferred_control_channels else {32: 128, 16: 128, 8: 128}

    use_noise_conditioning = any(k.startswith("unet.noise_embed.") for k in unet_state.keys())

    return {
        "latent_channels": latent_channels,
        "model_channels": model_channels,
        "channel_mult": channel_mult,
        "num_res_blocks": num_res_blocks,
        "attn_resolutions": attn_resolutions,
        "num_heads": int(hparams.get("num_heads", 8)),
        "dropout": 0.0,
        "init_res": init_res,
        "use_controlnet": use_controlnet,
        "control_channels": control_channels,
        "use_noise_conditioning": use_noise_conditioning,
    }


def load_model(
    model_checkpoint: str,
    vae_checkpoint: str,
    latent_stats_path: str,
    device: str = "cuda",
) -> SketchDenoiserModule:
    # Load VAE
    vae = VAE(latent_ch=4).to(device)
    if os.path.exists(vae_checkpoint):
        ckpt = torch.load(vae_checkpoint, map_location=device)
        vae_state = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
        vae.load_state_dict(vae_state)
    vae.eval()

    # Build UNet from checkpoint-derived architecture (works across model sizes).
    inferred_cfg = _infer_unet_config_from_checkpoint(model_checkpoint)
    unet = DenoisingUNet2DConditionModel(**inferred_cfg).to(device)
    print(
        "[benchmark] UNet config from checkpoint:",
        {
            "model_channels": inferred_cfg["model_channels"],
            "channel_mult": inferred_cfg["channel_mult"],
            "num_res_blocks": inferred_cfg["num_res_blocks"],
            "attn_resolutions": inferred_cfg["attn_resolutions"],
            "init_res": inferred_cfg["init_res"],
            "use_controlnet": inferred_cfg["use_controlnet"],
            "use_noise_conditioning": inferred_cfg["use_noise_conditioning"],
        },
    )

    model = SketchDenoiserModule.load_from_checkpoint(
        model_checkpoint,
        vae=vae,
        unet=unet,
        latent_stats_path=latent_stats_path,
        strict=True,
        map_location=device,
    ).to(device)
    print(
        "[benchmark] Conditioner from checkpoint:",
        {
            "use_controlnet": bool(getattr(model, "use_controlnet", False)),
            "control_encoder_type": str(getattr(model, "control_encoder_type", "unknown")),
            "dinov2_model_name": str(getattr(model.hparams, "dinov2_model_name", "n/a")),
            "train_dinov2_backbone": bool(getattr(model.hparams, "train_dinov2_backbone", False)),
        },
    )
    model.eval()
    return model


def _build_baseline_predictions(
    gt_norm: torch.Tensor,
    num_samples: int,
    baseline_mode: str,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Build deterministic baseline predictions with the same shape as model outputs.

    Supported baselines:
        - oracle_fg_mean_depth: predict one constant depth per item equal to the
          GT full-foreground mean depth, then convert back to normalized disparity.
    """
    baseline_mode = (baseline_mode or "").strip().lower()
    if baseline_mode not in {"oracle_fg_mean_depth"}:
        raise ValueError(f"Unsupported baseline_mode: {baseline_mode}")

    b, _, h, w = gt_norm.shape
    preds = torch.zeros((b, num_samples, 1, h, w), device=gt_norm.device, dtype=gt_norm.dtype)

    for idx in range(b):
        gt_map = gt_norm[idx, 0]
        fg_mask = gt_map > eps
        if fg_mask.sum() == 0:
            continue

        gt_depth = norm_disp_to_depth(gt_map)
        mean_depth = gt_depth[fg_mask].mean()
        mean_disp = depth_to_norm_disp(mean_depth.reshape(1)).reshape(())

        pred_map = torch.zeros_like(gt_map)
        pred_map[fg_mask] = mean_disp
        preds[idx, :, 0] = pred_map.unsqueeze(0).repeat(num_samples, 1, 1)

    return preds


def _fit_affine_scale_shift(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Least-squares fit for target ~= a * pred + b over masked pixels."""
    valid = mask & torch.isfinite(pred) & torch.isfinite(target)
    if valid.sum() < 2:
        a = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        b = target[mask].mean() if mask.sum() > 0 else torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        return a, b

    x = pred[valid]
    y = target[valid]
    x_mean = x.mean()
    y_mean = y.mean()
    x_var = ((x - x_mean) ** 2).mean()
    if x_var <= eps:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype), y_mean

    cov = ((x - x_mean) * (y - y_mean)).mean()
    a = cov / (x_var + eps)
    b = y_mean - a * x_mean
    return a, b


def _build_depth_anything_v2_predictions(
    sketches: torch.Tensor,
    gt_norm: torch.Tensor,
    num_samples: int,
    processor,
    depth_model,
    device: str,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Run Depth Anything V2 on sketch channel and align to GT disparity with affine fit."""
    b, _, h, w = sketches.shape
    preds = torch.zeros((b, num_samples, 1, h, w), device=gt_norm.device, dtype=gt_norm.dtype)

    sketch_np = sketches[:, 0].detach().float().cpu().numpy()
    images = []
    for idx in range(b):
        img = np.clip(sketch_np[idx], 0.0, 1.0)
        img = (1.0 - img) * 255.0
        img_u8 = img.astype(np.uint8)
        rgb = np.stack([img_u8, img_u8, img_u8], axis=-1)
        images.append(rgb)

    da_inputs = processor(images=images, return_tensors="pt")
    da_inputs = {k: v.to(device) for k, v in da_inputs.items()}

    with torch.no_grad():
        da_out = depth_model(**da_inputs)
        pred_depth = da_out.predicted_depth

    pred_depth_up = F.interpolate(
        pred_depth.unsqueeze(1),
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)

    pred_disp = 1.0 / torch.clamp(pred_depth_up, min=eps)

    for idx in range(b):
        gt_map = gt_norm[idx, 0]
        fg_mask = gt_map > eps
        if fg_mask.sum() == 0:
            continue

        a, b_shift = _fit_affine_scale_shift(pred_disp[idx], gt_map, fg_mask, eps=eps)
        aligned = torch.clamp(a * pred_disp[idx] + b_shift, 0.0, 1.0)
        aligned = torch.where(fg_mask, aligned, torch.zeros_like(aligned))
        preds[idx, :, 0] = aligned.unsqueeze(0).repeat(num_samples, 1, 1)

    return preds


def evaluate_model(
    model: SketchDenoiserModule | None,
    dataloader: DataLoader,
    num_steps: int = 50,
    cfg_scale: float = 3.0,
    use_ddim: bool = True,
    device: str = "cuda",
    num_samples: int = 1,
    predictions_dir: str | None = None,
    baseline_mode: str | None = None,
    depth_anything_model_id: str = "depth-anything/Depth-Anything-V2-Base-hf",
) -> pd.DataFrame:
    rows: List[Dict] = []
    total_items = 0
    skipped_items = 0

    num_samples = int(num_samples)
    if num_samples < 1:
        raise ValueError("num_samples must be >= 1")

    if predictions_dir is not None:
        os.makedirs(predictions_dir, exist_ok=True)

    baseline_mode_norm = (baseline_mode or "").strip().lower() if baseline_mode is not None else None
    depth_anything_processor = None
    depth_anything_model = None
    if baseline_mode_norm == "depth_anything_v2":
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except Exception as exc:
            raise RuntimeError(
                "Depth Anything V2 baseline requires transformers with AutoModelForDepthEstimation."
            ) from exc

        print(f"[benchmark] Loading Depth Anything V2 baseline: {depth_anything_model_id}")
        depth_anything_processor = AutoImageProcessor.from_pretrained(depth_anything_model_id)
        depth_anything_model = AutoModelForDepthEstimation.from_pretrained(depth_anything_model_id).to(device)
        depth_anything_model.eval()

    use_amp = device.startswith("cuda")
    for batch in tqdm(dataloader, total=len(dataloader), desc="Benchmark"):
        if batch is None:
            skipped_items += 1
            continue
        # Custom collate: batch is a list of dicts
        if isinstance(batch, list):
            items = [b for b in batch if b is not None]
        else:
            items = [batch]

        if not items:
            continue

        total_items += len(items)
        sketches = torch.stack([it["sketch"] for it in items], dim=0).to(device)
        gt_norm = torch.stack([it["gt_norm_depth"] for it in items], dim=0).to(device)

        # Support per-sample noise conditioning by grouping within the batch.
        noise_scalars = [float(it.get("noise_scalar", 0.0)) for it in items]
        unique_noise = sorted(set(noise_scalars))

        # Collect predictions per item: (B, S, 1, H, W)
        preds = torch.zeros(
            (len(items), num_samples, 1, sketches.shape[-2], sketches.shape[-1]),
            device=device,
            dtype=sketches.dtype,
        )

        with torch.no_grad():
            if baseline_mode is not None:
                if baseline_mode_norm == "oracle_fg_mean_depth":
                    preds = _build_baseline_predictions(gt_norm, num_samples=num_samples, baseline_mode=baseline_mode)
                elif baseline_mode_norm == "depth_anything_v2":
                    preds = _build_depth_anything_v2_predictions(
                        sketches=sketches,
                        gt_norm=gt_norm,
                        num_samples=num_samples,
                        processor=depth_anything_processor,
                        depth_model=depth_anything_model,
                        device=device,
                    )
                else:
                    raise ValueError(f"Unsupported baseline_mode: {baseline_mode}")
            else:
                if model is None:
                    raise ValueError("model must not be None when baseline_mode is not set")
                with torch.cuda.amp.autocast(enabled=use_amp):
                    for ns in unique_noise:
                        idxs = [i for i, v in enumerate(noise_scalars) if v == ns]
                        if not idxs:
                            continue
                        sub_sketch = sketches[idxs]
                        # Repeat each input num_samples times; stochasticity (if any)
                        # in the sampler/noise should yield different outputs.
                        sub_sketch_rep = sub_sketch.repeat_interleave(num_samples, dim=0)
                        sub_pred = model.inference(
                            sub_sketch_rep,
                            num_steps=num_steps,
                            cfg_scale=cfg_scale,
                            use_ddim=use_ddim,
                            noise_scalar=float(ns),
                            inpaint_known_pixels=True,
                            clamp_known_pixels=False
                        )
                        # Reshape back to (B_sub, S, 1, H, W)
                        sub_pred = sub_pred.reshape(len(idxs), num_samples, 1, sub_pred.shape[-2], sub_pred.shape[-1])
                        preds[idxs] = sub_pred

        for idx, item in enumerate(items):
            # For partial-depth cases, evaluate only on primary-unknown pixels.
            # NOTE: In occlusion-aware partial depth, the hint channel can be non-zero even when
            # the primary-visible edge at that pixel is unknown (hint is behind an occluder).
            # So we use primary_unknown_pixel_mask (if available) rather than (hint==0).
            mask = None
            if float(item.get("completion_ratio", 0.0)) > 0.0:
                m = item.get("primary_unknown_pixel_mask", None)
                if m is None:
                    m = item.get("unknown_pixel_mask", None)
                if isinstance(m, torch.Tensor):
                    mask = m.to(device)

            base_row = {
                "shape_id": item["shape_id"],
                "view_idx": item["view_idx"],
                "condition": item["condition"],
                "noise_level": float(item["noise_level"]),
                "noise_scalar": float(item.get("noise_scalar", 0.0)),
                "completion_ratio": float(item["completion_ratio"]),
                "known_edge_count": int(item.get("known_edge_count", 0)),
                "unknown_edge_count": int(item.get("unknown_edge_count", 0)),
                "known_pixel_count": int(item.get("known_pixel_count", 0)),
                "unknown_pixel_count": int(item.get("unknown_pixel_count", 0)),
                "primary_known_pixel_count": int(item.get("primary_known_pixel_count", 0)),
                "primary_unknown_pixel_count": int(item.get("primary_unknown_pixel_count", 0)),
                "hint_conflict_pixel_count": int(item.get("hint_conflict_pixel_count", 0)),
                "accidental_pixel_ratio": float(item["accidental_pixel_ratio"]),
                "accidental_pixel_ratio_all": float(item.get("accidental_pixel_ratio_all", item["accidental_pixel_ratio"])),
                "accidental_pixel_ratio_unknown": float(item.get("accidental_pixel_ratio_unknown", item["accidental_pixel_ratio"])),
                "visible_fg_pixels": int(item["visible_fg_pixels"]),
                "hidden_fg_fragments": int(item["hidden_fg_fragments"]),
                "visible_fg_pixels_all": int(item.get("visible_fg_pixels_all", item["visible_fg_pixels"])),
                "hidden_fg_fragments_all": int(item.get("hidden_fg_fragments_all", item["hidden_fg_fragments"])),
                "visible_fg_pixels_unknown": int(item.get("visible_fg_pixels_unknown", item["visible_fg_pixels"])),
                "hidden_fg_fragments_unknown": int(item.get("hidden_fg_fragments_unknown", item["hidden_fg_fragments"])),
                "num_edges": int(item["num_edges"]),
                "num_edges_all": int(item.get("num_edges_all", item["num_edges"])),
                "num_edges_unknown": int(item.get("num_edges_unknown", item["num_edges"])),
                "stroke_pixels": int(item["stroke_pixels"]),
                "topo_num_faces": int(item.get("topo_num_faces", -1)),
                "topo_num_edges": int(item.get("topo_num_edges", -1)),
                "topo_num_vertices": int(item.get("topo_num_vertices", -1)),
                "curve_complexity_score": float(item.get("curve_complexity_score", float("nan"))),
                "curve_edge_count": int(item.get("curve_edge_count", -1)),
                "curve_complexity_score_all": float(item.get("curve_complexity_score_all", float("nan"))),
                "curve_edge_count_all": int(item.get("curve_edge_count_all", -1)),
                "curve_complexity_score_unknown": float(item.get("curve_complexity_score_unknown", float("nan"))),
                "curve_edge_count_unknown": int(item.get("curve_edge_count_unknown", -1)),
            }

            for sample_idx in range(num_samples):
                pred_map = preds[idx, sample_idx, 0]
                metrics = disp_metrics_symmetric(pred_map, gt_norm[idx, 0], mask=mask)

                # Debug: region-specific MAE
                pk = item.get("primary_known_pixel_mask", None)
                pu = item.get("primary_unknown_pixel_mask", None)
                mae_primary_known = float("nan")
                mae_primary_unknown = float("nan")
                if isinstance(pk, torch.Tensor) and pk.sum() > 0:
                    mae_primary_known = disp_metrics_symmetric(pred_map, gt_norm[idx, 0], mask=pk.to(device)).get("mae", float("nan"))
                if isinstance(pu, torch.Tensor) and pu.sum() > 0:
                    mae_primary_unknown = disp_metrics_symmetric(pred_map, gt_norm[idx, 0], mask=pu.to(device)).get("mae", float("nan"))

                row = dict(base_row)
                row["sample_idx"] = int(sample_idx)
                row["mae_primary_known"] = float(mae_primary_known)
                row["mae_primary_unknown"] = float(mae_primary_unknown)

                # Optional on-disk recording of the raw prediction map for later
                # aggregation (mean/best over samples).
                if predictions_dir is not None:
                    safe_shape = str(item["shape_id"]).replace("/", "_")
                    fname = (
                        f"{safe_shape}__v{int(item['view_idx'])}__n{float(item['noise_level']):.4f}"
                        f"__c{float(item['completion_ratio']):.4f}__s{int(sample_idx)}.npz"
                    )
                    out_path = os.path.join(predictions_dir, fname)
                    np.savez_compressed(
                        out_path,
                        pred_norm_disp=pred_map.detach().float().cpu().numpy().astype(np.float16),
                    )
                    row["pred_path"] = out_path

                row.update(metrics)
                rows.append(row)

        # Count skipped items in a list batch (None entries)
        if isinstance(batch, list):
            skipped_items += sum(1 for b in batch if b is None)

    if total_items == 0:
        print("WARNING: No valid items were evaluated (all samples failed to render).")
    if skipped_items > 0:
        print(f"Skipped samples (None renders): {skipped_items}")

    return pd.DataFrame(rows)


def summarize_results(df: pd.DataFrame) -> dict:
    summary = {}
    if df.empty:
        return summary

    metrics = ["mae", "mae_range_norm", "absrel", "d1", "d2", "d3"]

    overall = {m: float(df[m].mean()) for m in metrics}
    summary["overall"] = overall

    # By condition
    by_condition = {}
    for cond, group in df.groupby("condition"):
        by_condition[cond] = {m: float(group[m].mean()) for m in metrics}
    summary["by_condition"] = by_condition

    # By completion ratio
    by_completion = {}
    for comp, group in df.groupby("completion_ratio"):
        by_completion[str(comp)] = {m: float(group[m].mean()) for m in metrics}
    summary["by_completion_ratio"] = by_completion

    # By noise level
    by_noise = {}
    for noise, group in df.groupby("noise_level"):
        by_noise[str(noise)] = {m: float(group[m].mean()) for m in metrics}
    summary["by_noise_level"] = by_noise

    # By accidental pixel ratio (quantile bins)
    try:
        df["accidental_bin"] = pd.qcut(df["accidental_pixel_ratio"], q=4, duplicates="drop")
        by_accidental = {}
        for bin_name, group in df.groupby("accidental_bin"):
            by_accidental[str(bin_name)] = {m: float(group[m].mean()) for m in metrics}
        summary["by_accidental_pixel_ratio_bin"] = by_accidental
    except Exception:
        summary["by_accidental_pixel_ratio_bin"] = {}

    # By shape complexity proxies (quantile bins)
    try:
        df["edge_bin"] = pd.qcut(df["num_edges"], q=4, duplicates="drop")
        by_edges = {}
        for bin_name, group in df.groupby("edge_bin"):
            by_edges[str(bin_name)] = {m: float(group[m].mean()) for m in metrics}
        summary["by_num_edges_bin"] = by_edges
    except Exception:
        summary["by_num_edges_bin"] = {}

    try:
        df["stroke_bin"] = pd.qcut(df["stroke_pixels"], q=4, duplicates="drop")
        by_strokes = {}
        for bin_name, group in df.groupby("stroke_bin"):
            by_strokes[str(bin_name)] = {m: float(group[m].mean()) for m in metrics}
        summary["by_stroke_pixels_bin"] = by_strokes
    except Exception:
        summary["by_stroke_pixels_bin"] = {}

    return summary


def save_summary(path: str, summary: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
