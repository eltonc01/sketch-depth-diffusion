"""Suite presets for benchmark runs.

Each suite returns a dict of overrides compatible with benchmark.run CLI args.
"""

from __future__ import annotations

from typing import Dict


def suite_ablation_encoder() -> Dict:
	"""Standard encoder ablation sweep defaults."""
	return {
		"noise_levels": [0.0],
		"completion_ratios": [0.0],
		"partial_depth_mode": "fast",
		"views_subset": [6, 7, 8, 9],
	}


def suite_partial_depth_curve() -> Dict:
	"""Sweep partial depth completion ratios."""
	return {
		"noise_levels": [0.0],
		"completion_ratios": [0.0, 0.1, 0.25, 0.5, 0.75, 0.85, 0.95],
		"partial_depth_mode": "occlusion_aware",
		"views_subset": [6, 7, 8, 9],
	}


def suite_difficulty_occlusion() -> Dict:
	"""Difficulty-focused sweep with noisy inputs and partial depth."""
	return {
		"noise_levels": [0.0, 0.02, 0.05],
		"completion_ratios": [0.0, 0.25, 0.5],
		"partial_depth_mode": "occlusion_aware",
		"views_subset": [6, 7, 8, 9],
	}


def get_suite(name: str) -> Dict:
	name = (name or "").strip().lower()
	if name in {"ablation", "ablation_encoder", "encoder"}:
		return suite_ablation_encoder()
	if name in {"partial", "partial_depth", "partial_depth_curve"}:
		return suite_partial_depth_curve()
	if name in {"difficulty", "difficulty_occlusion", "occlusion"}:
		return suite_difficulty_occlusion()
	raise ValueError(f"Unknown suite: {name}")
