# Reconstruction of a 3D wireframe from a single line drawing via generative depth estimation

### [Project Page](https://eltonc01.github.io/sketch-depth-diffusion/) | [Paper]() | [Weights](https://huggingface.co/eltoncao/sketch_depth/tree/main) | [Live Demo]()

[Reconstruction of a 3D wireframe from a single line drawing]()  
[Elton Cao](https://)<sup>1</sup>, [Hod Lipson](https://)<sup>1</sup><br>
<sup>1</sup>Columbia University

<img src="assets/teaser.gif">

## Setup

Preferred: create the conda environment from [environment.yml](environment.yml), since OpenCascade (`pythonocc-core`) is conda-managed.

```bash
conda env create -f environment.yml
conda activate sketch_recon
```

Pip-only install is possible for most Python dependencies:

```bash
python -m pip install -r requirements.txt
```

But `dataset_tools` and benchmark rendering paths that import `OCC` still require:

```bash
conda install -c conda-forge pythonocc-core=7.9.0
```

## Models Weights and Dataset

- Dataset repo: `eltoncao/wireframe-data`
- Model repo: `eltoncao/sketch_depth`
- Checkpoint manifest used by runners: [sketch_recon/config/checkpoints.json](sketch_recon/config/checkpoints.json)

Download weights & dataset:

- Model checkpoints (needed for inference/training):

```bash
python scripts/download_checkpoints.py --only models --model-variant dinov2_vast
```

- Dataset assets (only needed for benchmark/training, not the bundled inference demo):

```bash
python scripts/download_checkpoints.py --only dataset --dataset-name default --extract-dataset
```

## Inference Demo

Sample sketch masks included under [assets/demo_inputs](assets/demo_inputs), so you can run inference without generating/importing dataset assets first.

Run inference on bundled sketch PNGs:

```bash
python scripts/infer_demo.py \
  --model_variant dinov2_vast \
  --input_dir assets/demo_inputs \
  --output_dir demo_outputs/inference \
  --num_steps 20 \
  --cfg_scale 1.0
```

Outputs are written as:
- `<name>__pred_norm_disp.npz` (raw normalized disparity)
- `<name>__pred_norm_disp.png` (8-bit preview)

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

This runs a strict-clean benchmark preset and resolves checkpoints via the shared checkpoint manifest.

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
