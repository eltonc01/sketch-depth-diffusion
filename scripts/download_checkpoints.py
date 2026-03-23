#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sketch_recon.config.checkpoints import (
    available_dataset_names,
    ensure_parent_dir,
    resolve_dataset_paths,
    resolve_model_variant_paths,
)


def _extract_tar_gz(archive_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(str(archive_path), "r:gz") as tar:
        out_dir_resolved = out_dir.resolve()
        members = tar.getmembers()
        for member in members:
            target_path = (out_dir_resolved / member.name).resolve()
            if target_path != out_dir_resolved and out_dir_resolved not in target_path.parents:
                raise ValueError(f"Unsafe path in archive: {member.name}")
        tar.extractall(path=str(out_dir_resolved), members=members)


def download_dataset(
    manifest_path: str | None,
    dataset_name: str,
    token: str | None,
    extract_dataset: bool,
) -> None:
    dataset = resolve_dataset_paths(dataset_name=dataset_name, manifest_path=manifest_path)
    local_dir = Path(dataset["local_dir"])
    local_dir.mkdir(parents=True, exist_ok=True)

    archive_local = hf_hub_download(
        repo_id=dataset["repo_id"],
        repo_type=dataset["repo_type"],
        filename=dataset["filename"],
        revision=dataset.get("revision"),
        token=token,
        local_dir=str(local_dir),
    )
    archive_local_path = Path(archive_local)
    print(f"Downloaded dataset '{dataset_name}' archive to {archive_local_path}")

    if extract_dataset:
        _extract_tar_gz(archive_local_path, local_dir)
        print(f"Extracted dataset '{dataset_name}' archive under {local_dir}")


def download_all_datasets(manifest_path: str | None, token: str | None, extract_dataset: bool) -> None:
    names = available_dataset_names(manifest_path)
    if not names:
        raise ValueError("No dataset entries found in checkpoint manifest")
    for name in names:
        download_dataset(manifest_path, name, token, extract_dataset)


def download_model_variant(manifest_path: str | None, model_variant: str, token: str | None) -> None:
    spec = resolve_model_variant_paths(model_variant, manifest_path)
    repo_id = spec["repo_id"]
    repo_type = spec["repo_type"]
    revision = spec.get("revision")

    file_specs = [
        ("model_checkpoint", spec["model_checkpoint_in_repo"], spec["model_checkpoint"]),
        ("vae_checkpoint", spec["vae_checkpoint_in_repo"], spec["vae_checkpoint"]),
        ("latent_stats", spec["latent_stats_in_repo"], spec["latent_stats_path"]),
    ]

    cache_dir = Path(".hf_download_cache").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    for label, path_in_repo, local_path in file_specs:
        if not path_in_repo:
            raise ValueError(f"Missing path_in_repo for {label} in variant '{model_variant}'")

        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            filename=path_in_repo,
            revision=revision,
            token=token,
            local_dir=str(cache_dir),
        )
        ensure_parent_dir(local_path)
        shutil.copy2(downloaded_path, local_path)
        print(f"Saved {label} -> {local_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download datasets and model checkpoints using the checkpoint manifest.")
    parser.add_argument("--manifest", type=str, default=None, help="Path to checkpoint manifest JSON")
    parser.add_argument("--model-variant", type=str, default="dinov2_vast", help="Model variant to download")
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="default",
        help="Dataset entry name to download when using --only dataset",
    )
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        default=False,
        help="Download all dataset entries from the manifest",
    )
    parser.add_argument(
        "--only",
        type=str,
        default="all",
        choices=["all", "dataset", "models"],
        help="Limit operation to dataset or models",
    )
    parser.add_argument("--token", type=str, default=None, help="Optional HF token")
    parser.add_argument(
        "--extract-dataset",
        action="store_true",
        default=False,
        help="Extract dataset archive after download",
    )
    args = parser.parse_args()

    if args.only in {"all", "dataset"}:
        if args.all_datasets:
            download_all_datasets(args.manifest, args.token, bool(args.extract_dataset))
        else:
            download_dataset(args.manifest, args.dataset_name, args.token, bool(args.extract_dataset))

    if args.only in {"all", "models"}:
        download_model_variant(args.manifest, args.model_variant, args.token)

    print("Checkpoint and dataset download completed.")


if __name__ == "__main__":
    main()
