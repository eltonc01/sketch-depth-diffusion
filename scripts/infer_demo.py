#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.eval import load_model
from sketch_recon.config.checkpoints import resolve_model_variant_paths


def _device_str(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_sketch(path: Path, image_size: int, invert: bool, threshold: int | None) -> torch.Tensor:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")

    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)

    if threshold is not None:
        img = np.where(img >= threshold, 255, 0).astype(np.uint8)

    arr = img.astype(np.float32) / 255.0
    if invert:
        arr = 1.0 - arr

    return torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)


def _save_prediction(out_dir: Path, stem: str, pred: np.ndarray) -> None:
    pred = np.clip(pred.astype(np.float32), 0.0, 1.0)
    np.savez_compressed(out_dir / f"{stem}__pred_norm_disp.npz", pred_norm_disp=pred)

    pred_u8 = (pred * 255.0).round().astype(np.uint8)
    cv2.imwrite(str(out_dir / f"{stem}__pred_norm_disp.png"), pred_u8)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run inference on bundled sketch PNGs without requiring generated benchmark data."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="assets/demo_inputs",
        help="Directory of input sketch PNGs.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="demo_outputs/inference",
        help="Directory to save predictions.",
    )
    parser.add_argument(
        "--model_variant",
        type=str,
        default="dinov2_vast",
        help="Model variant key from sketch_recon/config/checkpoints.json.",
    )
    parser.add_argument(
        "--checkpoint_manifest",
        type=str,
        default=None,
        help="Optional custom checkpoint manifest path.",
    )
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--noise_scalar", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_candidates", type=int, default=1)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "cuda:0", "cuda:1"],
    )
    parser.add_argument(
        "--invert_sketch",
        action="store_true",
        default=False,
        help="Invert input sketches (use only if strokes are black and model expects white).",
    )
    parser.add_argument(
        "--binarize_threshold",
        type=int,
        default=None,
        help="Optional threshold in [0, 255] to binarize input sketch.",
    )
    args = parser.parse_args()

    if args.binarize_threshold is not None and not (0 <= int(args.binarize_threshold) <= 255):
        raise ValueError("--binarize_threshold must be in [0, 255]")

    device = _device_str(args.device)
    paths = resolve_model_variant_paths(args.model_variant, args.checkpoint_manifest)

    missing_ckpts = [
        p
        for p in [paths["model_checkpoint"], paths["vae_checkpoint"], paths["latent_stats_path"]]
        if not os.path.exists(p)
    ]
    if missing_ckpts:
        missing_str = "\n".join(f"  - {p}" for p in missing_ckpts)
        raise FileNotFoundError(
            "Missing checkpoint artifacts:\n"
            f"{missing_str}\n"
            "Download them first, e.g.:\n"
            "  python scripts/download_checkpoints.py --only models --model-variant "
            f"{args.model_variant}"
        )

    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = REPO_ROOT / input_dir
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    sketch_paths = sorted(input_dir.glob("*.png"))
    if not sketch_paths:
        raise FileNotFoundError(f"No PNG files found in: {input_dir}")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model variant: {args.model_variant}")
    print(f"Device: {device}")
    model = load_model(
        model_checkpoint=paths["model_checkpoint"],
        vae_checkpoint=paths["vae_checkpoint"],
        latent_stats_path=paths["latent_stats_path"],
        device=device,
    )

    use_amp = device.startswith("cuda")

    print(f"Found {len(sketch_paths)} sketch image(s) in {input_dir}")
    with torch.no_grad():
        for start in range(0, len(sketch_paths), int(args.batch_size)):
            batch_paths = sketch_paths[start : start + int(args.batch_size)]
            batch_tensors = [
                _load_sketch(
                    p,
                    image_size=int(args.image_size),
                    invert=bool(args.invert_sketch),
                    threshold=args.binarize_threshold,
                )
                for p in batch_paths
            ]
            batch = torch.stack(batch_tensors, dim=0).to(device=device, dtype=torch.float32)

            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model.inference(
                    batch,
                    num_steps=int(args.num_steps),
                    cfg_scale=float(args.cfg_scale),
                    use_ddim=True,
                    noise_scalar=float(args.noise_scalar),
                    inpaint_known_pixels=True,
                    clamp_known_pixels=False,
                    num_candidates=int(args.num_candidates),
                    select_by_structure=True,
                )

            pred_np = pred.detach().float().cpu().numpy()  # (B,1,H,W)
            for p, out in zip(batch_paths, pred_np):
                stem = p.stem
                _save_prediction(output_dir, stem, out[0])

    print(f"Saved predictions to: {output_dir}")


if __name__ == "__main__":
    main()
