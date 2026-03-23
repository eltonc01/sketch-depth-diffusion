from __future__ import annotations

import argparse
import os
import sys
from typing import List
import warnings

import torch

import pandas as pd
from torch.utils.data import DataLoader, Subset

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from benchmark.dataset import BenchmarkDatasetV2
from benchmark.eval import evaluate_model, load_model, save_summary, summarize_results
from benchmark.splits import get_test_split, load_shape_ids
from sketch_recon.config.checkpoints import resolve_model_variant_paths


def collate_identity(batch):
    return batch


def parse_float_list(values: List[str]) -> List[float]:
    out = []
    for v in values:
        out.extend([float(x) for x in v.split(",") if x != ""])
    return out


def validate_noise_policy(noise_levels: List[float], strict_clean: bool) -> List[float]:
    """Validate benchmark noise-level policy and return normalized list."""
    normalized = [float(x) for x in noise_levels]
    if strict_clean:
        noisy = [x for x in normalized if x > 0.0]
        if noisy:
            raise ValueError(
                "strict_clean is enabled but noise_levels contains non-zero values: "
                + ", ".join(str(x) for x in noisy)
            )
    return normalized


def _device_str(device_idx: int | None) -> str:
    if torch.cuda.is_available():
        return "cuda" if device_idx is None else f"cuda:{device_idx}"
    return "cpu"


def _make_dataloader(ds, batch_size: int, num_workers: int, prefetch_factor: int, mp_context: str):
    dl_kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=collate_identity,
        pin_memory=torch.cuda.is_available(),
    )
    if int(num_workers) > 0:
        dl_kwargs["persistent_workers"] = True
        dl_kwargs["prefetch_factor"] = int(prefetch_factor)
        dl_kwargs["multiprocessing_context"] = mp_context
    return DataLoader(ds, **dl_kwargs)


def _resolve_out_dir(results_root: str, requested_name: str) -> tuple[str, str]:
    requested_name = str(requested_name).strip() or "run"
    direct_path = os.path.join(results_root, requested_name)
    if not os.path.exists(direct_path):
        return requested_name, direct_path

    idx = 0
    while True:
        candidate_name = f"{requested_name}_{idx}"
        candidate_path = os.path.join(results_root, candidate_name)
        if not os.path.exists(candidate_path):
            return candidate_name, candidate_path
        idx += 1


def _sharded_eval_worker(
    rank: int,
    device_idx: int,
    world_size: int,
    out_dir: str,
    dataset,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    mp_context: str,
    model_checkpoint: str,
    vae_checkpoint: str,
    latent_stats_path: str,
    num_steps: int,
    cfg_scale: float,
    num_samples: int,
    save_predictions: bool,
    baseline_model: str | None,
    depth_anything_model_id: str,
):
    device = _device_str(device_idx)
    indices = list(range(rank, len(dataset), world_size))
    shard = Subset(dataset, indices)
    dl = _make_dataloader(shard, batch_size, num_workers, prefetch_factor, mp_context)
    model = None
    if baseline_model is None:
        model = load_model(
            model_checkpoint=model_checkpoint,
            vae_checkpoint=vae_checkpoint,
            latent_stats_path=latent_stats_path,
            device=device,
        )
    pred_dir = None
    if bool(save_predictions):
        pred_dir = os.path.join(out_dir, f"predictions_rank{rank}")
    shard_df = evaluate_model(
        model,
        dl,
        num_steps=num_steps,
        cfg_scale=cfg_scale,
        use_ddim=True,
        device=device,
        num_samples=int(num_samples),
        predictions_dir=pred_dir,
        baseline_mode=baseline_model,
        depth_anything_model_id=depth_anything_model_id,
    )
    shard_df.to_csv(os.path.join(out_dir, f"results_rank{rank}.csv"), index=False)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--zip_dir", type=str, default="abc/zip_clean")
    parser.add_argument("--step_dir", type=str, default="abc/step")
    parser.add_argument("--test_ids_file", type=str, default=None)
    parser.add_argument(
        "--max_shapes",
        type=int,
        default=None,
        help="Optional cap on number of shapes to evaluate (useful for quick logic checks). Default: no limit.",
    )

    parser.add_argument("--noise_levels", type=str, nargs="*", default=["0.0"],
                        help="Comma-separated list(s) of noise levels")
    parser.add_argument("--completion_ratios", type=str, nargs="*", default=["0.0,0.25,0.5"],
                        help="Comma-separated list(s) of completion ratios")
    parser.add_argument("--partial_depth_mode", type=str, default="occlusion_aware", choices=["fast", "occlusion_aware"])

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    _bool_action = getattr(argparse, "BooleanOptionalAction", None)
    if _bool_action is not None:
        parser.add_argument(
            "--use_cache",
            action=_bool_action,
            default=True,
            help="Enable on-disk render cache to reuse clean/noisy renders.",
        )
    else:
        parser.add_argument("--use_cache", action="store_true", default=True)
        parser.add_argument("--no_use_cache", action="store_false", dest="use_cache")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="benchmark/render_cache",
        help="Directory for render cache files.",
    )
    parser.add_argument(
        "--mp_context",
        type=str,
        default="spawn",
        choices=["spawn", "forkserver", "fork"],
        help="Multiprocessing context for DataLoader workers. Use spawn/forkserver for OCC safety.",
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=2,
        help="Prefetch factor for DataLoader workers (only when num_workers>0).",
    )

    parser.add_argument(
        "--devices",
        type=str,
        default=None,
        help="Comma-separated CUDA device indices for sharded multi-GPU eval (e.g. '0,1'). Default: single GPU.",
    )

    parser.add_argument("--model_checkpoint", type=str, default=None)
    parser.add_argument("--vae_checkpoint", type=str, default=None)
    parser.add_argument("--latent_stats_path", type=str, default=None)
    parser.add_argument(
        "--model_variant",
        type=str,
        default="dinov2_vast",
        help="Model variant key in checkpoint manifest used to resolve checkpoint paths.",
    )
    parser.add_argument(
        "--checkpoint_manifest",
        type=str,
        default=None,
        help="Optional path to checkpoint manifest JSON. Defaults to sketch_recon/config/checkpoints.json.",
    )
    parser.add_argument(
        "--artifact_manifest",
        dest="checkpoint_manifest",
        type=str,
        help=argparse.SUPPRESS,
    )

    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument(
        "--baseline_model",
        type=str,
        default=None,
        choices=["oracle_fg_mean_depth", "depth_anything_v2"],
        help="Optional non-learned baseline to evaluate instead of loading a model checkpoint.",
    )
    parser.add_argument(
        "--depth_anything_model_id",
        type=str,
        default="depth-anything/Depth-Anything-V2-Base-hf",
        help="Hugging Face model id used when --baseline_model depth_anything_v2.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of stochastic diffusion samples per rendered input. Each sample is recorded as its own row.",
    )
    if _bool_action is not None:
        parser.add_argument(
            "--strict_clean",
            action=_bool_action,
            default=True,
            help="Paper release benchmark is clean-only; non-zero noise levels are rejected.",
        )
    else:
        parser.add_argument("--strict_clean", action="store_true", default=True)
        parser.add_argument("--no_strict_clean", action="store_false", dest="strict_clean")
    if _bool_action is not None:
        parser.add_argument(
            "--save_predictions",
            action=_bool_action,
            default=False,
            help="If enabled, saves each predicted norm-disparity map to out_dir/predictions for later mean/best aggregation.",
        )
    else:
        parser.add_argument("--save_predictions", action="store_true", default=False)
        parser.add_argument("--no_save_predictions", action="store_false", dest="save_predictions")
    parser.add_argument("--run_name", type=str, default="run")
    parser.add_argument("--views_subset", type=str, default="6,7,8,9",
                        help="Comma-separated list of pose indices to use (e.g., '6,7,8,9').")
    parser.add_argument(
        "--render_retry_random",
        type=int,
        default=2,
        help="If a render fails for a fixed view, retry with this many deterministic random viewpoints.",
    )
    parser.add_argument(
        "--render_retry_seed",
        type=int,
        default=0,
        help="Base seed for deterministic random viewpoint retries (for reproducibility + caching).",
    )
    parser.add_argument("--suite", type=str, default=None,
                        help="Preset suite name: ablation_encoder, partial_depth_curve, difficulty_occlusion")

    parser.add_argument("--use_imperfect_to_perfect", action="store_true", default=True)

    args = parser.parse_args()

    if not bool(args.strict_clean):
        raise ValueError(
            "Paper release benchmark enforces clean-only evaluation. "
            "Remove --no-strict-clean and keep noise_levels at 0.0."
        )

    # pythonOCC / OpenCASCADE is not fork-safe; if the user requests DataLoader workers,
    # prefer spawn/forkserver to avoid inheriting OCC state into child processes.
    if args.num_workers and args.num_workers > 0:
        if args.mp_context == "fork":
            warnings.warn(
                "mp_context=fork can break pythonOCC (render failures or hangs). Use spawn/forkserver instead.",
                stacklevel=1,
            )
        try:
            torch.multiprocessing.set_start_method(args.mp_context, force=True)
        except RuntimeError:
            # Start method might already be set by the environment.
            pass

    # Normalize paths relative to repo root if needed
    if not os.path.isabs(args.zip_dir):
        args.zip_dir = os.path.join(REPO_ROOT, args.zip_dir)
    if not os.path.isabs(args.step_dir):
        args.step_dir = os.path.join(REPO_ROOT, args.step_dir)
    if not os.path.isabs(args.cache_dir):
        args.cache_dir = os.path.join(REPO_ROOT, args.cache_dir)

    if args.test_ids_file:
        shape_ids = load_shape_ids(args.test_ids_file)
    else:
        shape_ids = get_test_split(args.zip_dir)

    if args.baseline_model is None:
        if not (args.model_checkpoint and args.vae_checkpoint and args.latent_stats_path):
            resolved = resolve_model_variant_paths(
                model_variant=args.model_variant,
                manifest_path=args.checkpoint_manifest,
            )
            args.model_checkpoint = args.model_checkpoint or resolved["model_checkpoint"]
            args.vae_checkpoint = args.vae_checkpoint or resolved["vae_checkpoint"]
            args.latent_stats_path = args.latent_stats_path or resolved["latent_stats_path"]

        missing = [
            name
            for name, value in [
                ("--model_checkpoint", args.model_checkpoint),
                ("--vae_checkpoint", args.vae_checkpoint),
                ("--latent_stats_path", args.latent_stats_path),
            ]
            if not value
        ]
        if missing:
            raise ValueError(
                "Model evaluation requires explicit checkpoint paths. Missing: "
                + ", ".join(missing)
            )

    print(len(shape_ids))

    if args.max_shapes is not None:
        max_shapes = int(args.max_shapes)
        if max_shapes > 0:
            shape_ids = shape_ids[:max_shapes]
        elif max_shapes == 0:
            shape_ids = []
        else:
            warnings.warn("--max_shapes was provided but < 0; ignoring cap.", stacklevel=1)

    noise_levels = parse_float_list(args.noise_levels)
    completion_ratios = parse_float_list(args.completion_ratios)

    if args.suite:
        from benchmark.suites import get_suite
        preset = get_suite(args.suite)
        noise_levels = preset.get("noise_levels", noise_levels)
        completion_ratios = preset.get("completion_ratios", completion_ratios)
        if args.partial_depth_mode == "fast":
            args.partial_depth_mode = preset.get("partial_depth_mode", args.partial_depth_mode)
        if args.views_subset is None:
            args.views_subset = ",".join(str(v) for v in preset.get("views_subset", []))

    noise_levels = validate_noise_policy(noise_levels, bool(args.strict_clean))

    # Minimal args object for TopoMapper (renderer)
    import argparse as _arg
    render_args = _arg.Namespace()
    render_args.width = 256
    render_args.height = 256
    render_args.line_diameter = 0.02
    render_args.tol = 1e-4
    render_args.fov = 30
    render_args.focus = 0
    render_args.apply_jitter = False
    render_args.apply_perlin = False
    render_args.jitter_strength = 0.0
    render_args.perlin_strength = 0.0
    render_args.perlin_scale = 2.0
    render_args.enable_partial_depth = False
    render_args.location = [1.0, 1.0, 1.0]
    render_args.direction = [1.0, 1.0, 1.0]
    render_args.pose = None

    views_subset = None
    if args.views_subset:
        views_subset = [int(x) for x in args.views_subset.split(",") if x != ""]

    dataset = BenchmarkDatasetV2(
        shape_ids=shape_ids,
        step_dir=args.step_dir,
        args=render_args,
        noise_levels=noise_levels,
        completion_ratios=completion_ratios,
        use_imperfect_to_perfect=args.use_imperfect_to_perfect,
        partial_depth_mode=args.partial_depth_mode,
        views_subset=views_subset,
        use_cache=bool(args.use_cache),
        cache_dir=args.cache_dir,
        render_retry_random=int(args.render_retry_random),
        render_retry_seed=int(args.render_retry_seed),
    )

    print(f"zip_dir: {args.zip_dir}")
    print(f"step_dir: {args.step_dir}")
    print(f"Benchmark samples: {len(dataset)} (shapes={len(shape_ids)}, views={len(views_subset or [6,7,8,9])}, noise_levels={noise_levels}, completion_ratios={completion_ratios})")
    if len(shape_ids) == 0:
        print("WARNING: No shapes found. Check that zip_dir points to abc/zip_clean with .npz files.")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # Device parsing
    devices: List[int] = []
    if args.devices:
        devices = [int(x) for x in args.devices.split(",") if x != ""]
    if devices and not torch.cuda.is_available():
        raise RuntimeError("--devices was provided but CUDA is not available")

    if len(devices) > 1 and int(args.num_workers) > 0:
        warnings.warn(
            "Using multi-GPU sharding together with DataLoader num_workers>0 will create nested multiprocessing. "
            "If you hit instability, set --num_workers 0.",
            stacklevel=1,
        )

    results_root = os.path.join(REPO_ROOT, "benchmark", "results")
    resolved_run_name, out_dir = _resolve_out_dir(results_root, args.run_name)
    os.makedirs(out_dir, exist_ok=True)
    if resolved_run_name != args.run_name:
        print(f"run_name '{args.run_name}' already exists; using '{resolved_run_name}'")

    # Write manifest (the planned conditions grid)
    manifest_rows = []
    for shape_id, view_idx, noise_level, completion_ratio, condition in dataset.samples:
        manifest_rows.append({
            "shape_id": shape_id,
            "view_idx": view_idx,
            "noise_level": float(noise_level),
            "completion_ratio": float(completion_ratio),
            "condition": condition,
            "partial_depth_mode": args.partial_depth_mode,
        })
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(os.path.join(out_dir, "manifest.csv"), index=False)

    if len(devices) <= 1:
        # Single-device run (default CUDA/CPU OR explicit single --devices id)
        device = _device_str(devices[0] if len(devices) == 1 else None)
        dataloader = _make_dataloader(dataset, args.batch_size, args.num_workers, args.prefetch_factor, args.mp_context)
        model = None
        if args.baseline_model is None:
            model = load_model(
                model_checkpoint=args.model_checkpoint,
                vae_checkpoint=args.vae_checkpoint,
                latent_stats_path=args.latent_stats_path,
                device=device,
            )
        pred_dir = os.path.join(out_dir, "predictions") if bool(args.save_predictions) else None
        df = evaluate_model(
            model,
            dataloader,
            num_steps=args.num_steps,
            cfg_scale=args.cfg_scale,
            use_ddim=True,
            device=device,
            num_samples=int(args.num_samples),
            predictions_dir=pred_dir,
            baseline_mode=args.baseline_model,
            depth_anything_model_id=args.depth_anything_model_id,
        )
    else:
        # Multi-GPU sharded run: each process evaluates a disjoint subset of dataset indices.
        import torch.multiprocessing as mp

        ctx = mp.get_context("spawn")
        procs = []
        world_size = len(devices)
        for rank, device_idx in enumerate(devices):
            p = ctx.Process(
                target=_sharded_eval_worker,
                args=(
                    rank,
                    device_idx,
                    world_size,
                    out_dir,
                    dataset,
                    int(args.batch_size),
                    int(args.num_workers),
                    int(args.prefetch_factor),
                    args.mp_context,
                    args.model_checkpoint,
                    args.vae_checkpoint,
                    args.latent_stats_path,
                    int(args.num_steps),
                    float(args.cfg_scale),
                    int(args.num_samples),
                    bool(args.save_predictions),
                    args.baseline_model,
                    args.depth_anything_model_id,
                ),
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"Multi-GPU worker exited with code {p.exitcode}")

        # Merge shard outputs
        shard_paths = [os.path.join(out_dir, f"results_rank{r}.csv") for r in range(world_size)]
        shard_dfs = [pd.read_csv(p) for p in shard_paths if os.path.exists(p)]
        df = pd.concat(shard_dfs, ignore_index=True) if shard_dfs else pd.DataFrame()

    results_path = os.path.join(out_dir, "results.csv")
    df.to_csv(results_path, index=False)

    summary = summarize_results(df)
    save_summary(os.path.join(out_dir, "summary.json"), summary)

    print(f"Saved results to {results_path}")
    if df.empty:
        print("WARNING: results are empty. Check render errors above or OCC setup.")


if __name__ == "__main__":
    main()
