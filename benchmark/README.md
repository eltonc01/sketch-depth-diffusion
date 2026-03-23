# Benchmark

## Quick smoke

```bash
cd paper-wireframe-depth
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python -m py_compile benchmark/run.py benchmark/eval.py benchmark/dataset.py
python benchmark/run.py --help > /dev/null
python benchmark/run.py --baseline_model oracle_fg_mean_depth --strict_clean --noise_levels 0.0 --completion_ratios 0.0 --max_shapes 0 --num_workers 0 --batch_size 1 --run_name smoke_benchmark_quick
```

## Policy

- Paper release benchmark enforces noise level 0.0.
- Noisy codepaths are retained but not exercised by default paper commands.
- strict_clean rejects any non-zero noise level.

## Outputs

- benchmark/results/<run_name>/results.csv
- benchmark/results/<run_name>/summary.json
