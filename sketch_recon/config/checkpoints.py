from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_manifest_path() -> Path:
    return repo_root() / "sketch_recon" / "config" / "checkpoints.json"


def load_checkpoint_manifest(manifest_path: str | None = None) -> dict[str, Any]:
    path = Path(manifest_path).expanduser().resolve() if manifest_path else default_manifest_path()
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "models" not in data or not isinstance(data["models"], dict):
        raise ValueError("Checkpoint manifest must define a 'models' mapping")
    return data


def available_model_variants(manifest_path: str | None = None) -> list[str]:
    manifest = load_checkpoint_manifest(manifest_path)
    return sorted(manifest.get("models", {}).keys())


def _resolve_local_path(local_path: str) -> str:
    path = Path(local_path)
    if path.is_absolute():
        return str(path)
    return str((repo_root() / path).resolve())


def resolve_model_variant_paths(model_variant: str, manifest_path: str | None = None) -> dict[str, Any]:
    manifest = load_checkpoint_manifest(manifest_path)
    variants = manifest.get("models", {})
    if model_variant not in variants:
        known = ", ".join(sorted(variants.keys()))
        raise KeyError(f"Unknown model variant '{model_variant}'. Known variants: {known}")

    variant = variants[model_variant]
    files = variant.get("files", {})

    def _entry(name: str) -> dict[str, Any]:
        if name not in files:
            raise KeyError(f"Model variant '{model_variant}' missing files.{name} entry")
        entry = files[name]
        if "local_path" not in entry:
            raise KeyError(f"Model variant '{model_variant}' files.{name} missing local_path")
        return entry

    model_ckpt = _entry("model_checkpoint")
    vae_ckpt = _entry("vae_checkpoint")
    latent_stats = _entry("latent_stats")

    return {
        "model_variant": model_variant,
        "repo_id": variant.get("repo_id"),
        "repo_type": variant.get("repo_type", "model"),
        "revision": variant.get("revision"),
        "model_checkpoint": _resolve_local_path(str(model_ckpt["local_path"])),
        "vae_checkpoint": _resolve_local_path(str(vae_ckpt["local_path"])),
        "latent_stats_path": _resolve_local_path(str(latent_stats["local_path"])),
        "model_checkpoint_in_repo": model_ckpt.get("path_in_repo"),
        "vae_checkpoint_in_repo": vae_ckpt.get("path_in_repo"),
        "latent_stats_in_repo": latent_stats.get("path_in_repo"),
    }


def resolve_dataset_manifest(manifest_path: str | None = None) -> dict[str, Any]:
    manifest = load_checkpoint_manifest(manifest_path)
    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("Checkpoint manifest must define 'dataset'")

    local_dir = dataset.get("local_dir", "abc")
    return {
        "repo_id": dataset.get("repo_id"),
        "repo_type": dataset.get("repo_type", "dataset"),
        "revision": dataset.get("revision"),
        "filename": dataset.get("filename"),
        "local_dir": _resolve_local_path(str(local_dir)),
    }


def available_dataset_names(manifest_path: str | None = None) -> list[str]:
    manifest = load_checkpoint_manifest(manifest_path)
    datasets = manifest.get("datasets")
    if isinstance(datasets, dict) and datasets:
        return sorted(datasets.keys())
    if isinstance(manifest.get("dataset"), dict):
        return ["default"]
    return []


def resolve_dataset_paths(dataset_name: str = "default", manifest_path: str | None = None) -> dict[str, Any]:
    manifest = load_checkpoint_manifest(manifest_path)

    datasets = manifest.get("datasets")
    if isinstance(datasets, dict) and datasets:
        if dataset_name not in datasets:
            known = ", ".join(sorted(datasets.keys()))
            raise KeyError(f"Unknown dataset '{dataset_name}'. Known datasets: {known}")
        dataset = datasets[dataset_name]
    else:
        dataset = manifest.get("dataset")
        if not isinstance(dataset, dict):
            raise ValueError("Checkpoint manifest must define either 'dataset' or 'datasets'")

    local_dir = dataset.get("local_dir", "abc")
    return {
        "name": dataset_name,
        "repo_id": dataset.get("repo_id"),
        "repo_type": dataset.get("repo_type", "dataset"),
        "revision": dataset.get("revision"),
        "filename": dataset.get("filename"),
        "local_dir": _resolve_local_path(str(local_dir)),
    }


def ensure_parent_dir(path: str) -> None:
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def local_checkpoints_exist(model_variant: str, manifest_path: str | None = None) -> bool:
    paths = resolve_model_variant_paths(model_variant, manifest_path)
    return all(os.path.exists(paths[key]) for key in ["model_checkpoint", "vae_checkpoint", "latent_stats_path"])
