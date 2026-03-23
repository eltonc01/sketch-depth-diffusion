# Paper Wireframe Depth

Public release of sketch-conditioned diffusion for wireframe depth estimation.

## What This Reproduces

This repository reproduces the paper-core workflow:
- sketch-conditioned diffusion inference over VAE latents
- benchmark evaluation under clean, deterministic settings
- training launch path for the dinov2_vast paper baseline configuration

v1.0 scope excludes the interface stack and broader internal research variants.

## Quickstart (Under 5 Commands)

```bash
cd paper-wireframe-depth
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python -m pip install -r requirements.txt
python scripts/download_checkpoints.py --extract-dataset --model-variant dinov2_vast
python -m py_compile sketch_recon/training/train_diffusion.py benchmark/run.py benchmark/eval.py benchmark/dataset.py
python benchmark/run.py --strict_clean --noise_levels 0.0 --completion_ratios 0.0 --views_subset 6 --max_shapes 2 --num_samples 1 --save_predictions --num_steps 20 --cfg_scale 1.0 --model_variant dinov2_vast --run_name infer_demo_dinov2_vast_steps20_cfg1.0
```

## Pretrained Models and Dataset

- Dataset repo (kept as-is): `eltoncao/wireframe-data`
- Model repo (organized variants): `eltoncao/sketch_depth`
- Checkpoint manifest used by runners: `sketch_recon/config/checkpoints.json`

If you need a different local layout, pass explicit path overrides to run scripts or update the manifest.

## Inference Demo

```bash
cd paper-wireframe-depth
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python benchmark/run.py \
	--strict_clean \
	--noise_levels 0.0 \
	--completion_ratios 0.0 \
	--views_subset 6 \
	--max_shapes 2 \
	--num_samples 1 \
	--save_predictions \
	--num_steps 20 \
	--cfg_scale 1.0 \
	--model_variant dinov2_vast \
	--run_name infer_demo_dinov2_vast_steps20_cfg1.0
```

Default behavior:
- tiny fixed input set (`--max_shapes 2`, `--views_subset 6`)
- deterministic output directory naming under `benchmark/results/`
- predictions written for quick visual checks

## Benchmark Command

```bash
cd paper-wireframe-depth
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python benchmark/run.py \
	--strict_clean \
	--noise_levels 0.0 \
	--completion_ratios 0.0 \
	--suite difficulty_occlusion \
	--model_variant dinov2_vast \
	--run_name paper_eval_dinov2_vast
```

This runs a strict-clean paper benchmark preset and resolves checkpoints via the shared checkpoint manifest.

## Training Command

```bash
cd paper-wireframe-depth
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python sketch_recon/training/train_diffusion.py \
	--data_mode clean \
	--use_controlnet \
	--control_encoder dinov2 \
	--batch_size 16 \
	--epochs 100 \
	--precision bf16-mixed \
	--model_variant dinov2_vast
```

This command uses manifest-resolved VAE/latent-stat paths unless you override them.
