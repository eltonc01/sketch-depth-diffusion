# Reconstruction of a 3D wireframe from a single line drawing

### [Project Page]() | [Paper]() | [Weights](https://huggingface.co/eltoncao/sketch_depth/tree/main) | [Live Demo]()

[Reconstruction of a 3D wireframe from a single line drawing]()  
 [Elton Cao](https://)<sup>1</sup>, [Hod Lipson](https://)<sup>1</sup><br>
 <sup>1</sup>Columbia University

## Requirements

```bash
python -m pip install -r requirements.txt
```

## Models Weights and Dataset

- Dataset repo (kept as-is): `eltoncao/wireframe-data`
- Model repo (organized variants): `eltoncao/sketch_depth`
- Checkpoint manifest used by runners: `sketch_recon/config/checkpoints.json`

If you need a different local layout, pass explicit path overrides to run scripts or update the manifest.

## Inference Demo

```bash
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

## Run Benchmark

```bash
python benchmark/run.py \
	--strict_clean \
	--noise_levels 0.0 \
	--completion_ratios 0.0 \
	--suite difficulty_occlusion \
	--model_variant dinov2_vast \
	--run_name paper_eval_dinov2_vast
```

This runs a strict-clean paper benchmark preset and resolves checkpoints via the shared checkpoint manifest.

## Training

```bash
python sketch_recon/training/train_diffusion.py \
	--data_mode clean \
	--use_controlnet \
	--control_encoder dinov2 \
	--batch_size 192 \
	--epochs 100 \
	--precision bf16-mixed \
	--model_variant dinov2_vast
```
