"""Render cache utilities for benchmark rendering.

Caches clean renders and noisy sketch masks to disk so repeated evaluations
can reuse expensive OpenCASCADE render outputs across noise/partial settings.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from typing import Any, Dict, Optional

import numpy as np


CACHE_VERSION = 2


def _load_topomapper_class():
	from dataset_tools.topomapper import TopoMapper
	return TopoMapper


def _edge_curve_costs_from_topomapper(topo) -> np.ndarray:
	"""Compute per-edge curve-type costs aligned with TopoMapper edge indices."""
	from OCC.Core.BRep import BRep_Tool
	from OCC.Core.Geom import (
		Geom_Line,
		Geom_Circle,
		Geom_Ellipse,
		Geom_BSplineCurve,
		Geom_BezierCurve,
	)

	costs = {
		"line": 1.0,
		"circle": 5.0,
		"ellipse": 8.0,
		"bspline": 10.0,
		"bezier": 10.0,
		"other": 5.0,
	}

	max_idx = 0
	for e in topo.all_edges.values():
		try:
			max_idx = max(max_idx, int(e.index))
		except Exception:
			continue

	arr = np.zeros((max_idx + 1,), dtype=np.float32)
	for e in topo.all_edges.values():
		try:
			edge_idx = int(e.index)
			edge = e.edge
		except Exception:
			continue
		try:
			curve_result = BRep_Tool.Curve(edge)
			if curve_result is None or curve_result[0] is None:
				continue
			curve_handle = curve_result[0]
		except Exception:
			arr[edge_idx] = costs["other"]
			continue

		try:
			if Geom_Line.DownCast(curve_handle) is not None:
				arr[edge_idx] = costs["line"]
			elif Geom_Circle.DownCast(curve_handle) is not None:
				arr[edge_idx] = costs["circle"]
			elif Geom_Ellipse.DownCast(curve_handle) is not None:
				arr[edge_idx] = costs["ellipse"]
			elif Geom_BSplineCurve.DownCast(curve_handle) is not None:
				arr[edge_idx] = costs["bspline"]
			elif Geom_BezierCurve.DownCast(curve_handle) is not None:
				arr[edge_idx] = costs["bezier"]
			else:
				arr[edge_idx] = costs["other"]
		except Exception:
			arr[edge_idx] = costs["other"]

	return arr


def _json_hash(payload: Dict[str, Any]) -> str:
	data = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
	return hashlib.sha1(data).hexdigest()


def _seed_everything(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed % (2**32 - 1))


def _safe_np_savez(path: str, data: Dict[str, Any]) -> None:
	os.makedirs(os.path.dirname(path), exist_ok=True)
	tmp_fd, tmp_path = tempfile.mkstemp(suffix=".npz", dir=os.path.dirname(path))
	os.close(tmp_fd)
	try:
		np.savez_compressed(tmp_path, **data)
		os.replace(tmp_path, path)
	finally:
		if os.path.exists(tmp_path):
			try:
				os.remove(tmp_path)
			except OSError:
				pass


def _pack_object(x: Any) -> np.ndarray:
	return np.array(x, dtype=object)


def _unpack_object(x: np.ndarray) -> Any:
	return x.tolist()


def _render_signature(args) -> Dict[str, Any]:
	return {
		"width": int(args.width),
		"height": int(args.height),
		"line_diameter": float(args.line_diameter),
		"tol": float(args.tol),
		"fov": float(args.fov),
		"focus": float(args.focus),
		"location": [float(v) for v in args.location],
		"direction": [float(v) for v in args.direction],
	}


class RenderCache:
	def __init__(self, cache_dir: str, enabled: bool = True) -> None:
		self.cache_dir = cache_dir
		self.enabled = enabled

	def _path(self, kind: str, shape_id: str, view_idx: int, key_hash: str) -> str:
		name = f"{shape_id}_view{view_idx}_{key_hash}.npz"
		return os.path.join(self.cache_dir, kind, name)

	def _load_npz(self, path: str) -> Optional[Dict[str, Any]]:
		if not os.path.exists(path):
			return None
		with np.load(path, allow_pickle=True) as z:
			data = {k: z[k] for k in z.files}
		return data

	def _save_npz(self, path: str, data: Dict[str, Any]) -> None:
		_safe_np_savez(path, data)

	def get_clean_render(
		self,
		shape,
		shape_id: str,
		view_idx: int,
		args,
	) -> Optional[Dict[str, Any]]:
		sig = _render_signature(args)
		key = {"v": CACHE_VERSION, "kind": "clean", "shape_id": shape_id, "view_idx": int(view_idx), "sig": sig}
		key_hash = _json_hash(key)
		path = self._path("clean", shape_id, view_idx, key_hash)

		if self.enabled:
			cached = self._load_npz(path)
			if cached is not None:
				edge_curve_costs = cached.get("edge_curve_costs", None)
				# Backfill older cache entries that were written without per-edge costs.
				if edge_curve_costs is None:
					try:
						TopoMapper = _load_topomapper_class()
						topo = TopoMapper(shape, args)
						edge_curve_costs = _edge_curve_costs_from_topomapper(topo)
						to_save = dict(cached)
						to_save["edge_curve_costs"] = edge_curve_costs
						self._save_npz(path, to_save)
					except Exception:
						edge_curve_costs = None
				return {
					"depth_buffer": cached["depth_buffer"],
					"edge_buffer": cached["edge_buffer"],
					"adjacency_list": _unpack_object(cached["adjacency_list"]),
					"coords": _unpack_object(cached["coords"]),
					"depth_fragments": _unpack_object(cached["depth_fragments"]),
					"edge_fragments": _unpack_object(cached["edge_fragments"]),
					"perfect_edges": _unpack_object(cached["perfect_edges"]),
					"edge_curve_costs": edge_curve_costs,
				}

		# Render clean
		prev = (args.apply_jitter, args.apply_perlin, args.jitter_strength, args.perlin_strength, args.enable_partial_depth)
		args.apply_jitter = False
		args.apply_perlin = False
		args.jitter_strength = 0.0
		args.perlin_strength = 0.0
		args.enable_partial_depth = False

		TopoMapper = _load_topomapper_class()
		topo = TopoMapper(shape, args)
		edge_curve_costs = _edge_curve_costs_from_topomapper(topo)
		render_data = topo.generate_pairs_to_memory(
			args.width, args.height, line_diameter=args.line_diameter, render_as_tube=True
		)

		args.apply_jitter, args.apply_perlin, args.jitter_strength, args.perlin_strength, args.enable_partial_depth = prev

		if render_data is None:
			return None

		render_data["edge_curve_costs"] = edge_curve_costs

		if self.enabled:
			to_save = {
				"depth_buffer": render_data["depth_buffer"],
				"edge_buffer": render_data["edge_buffer"],
				"adjacency_list": _pack_object(render_data["adjacency_list"]),
				"coords": _pack_object(render_data.get("coords", [])),
				"depth_fragments": _pack_object(render_data.get("depth_fragments", [])),
				"edge_fragments": _pack_object(render_data.get("edge_fragments", [])),
				"perfect_edges": _pack_object(render_data.get("perfect_edges", [])),
				"edge_curve_costs": edge_curve_costs,
				"meta": _pack_object(key),
			}
			self._save_npz(path, to_save)

		return render_data

	def get_noisy_sketch_mask(
		self,
		shape,
		shape_id: str,
		view_idx: int,
		args,
		noise_level: float,
		noise_mode: Optional[str] = None,
	) -> Optional[np.ndarray]:
		sig = _render_signature(args)

		if noise_mode is None:
			# Deterministic noise mode based on key hash
			mode_key = {"shape_id": shape_id, "view_idx": int(view_idx), "noise_level": float(noise_level)}
			mode_hash = int(_json_hash(mode_key), 16)
			noise_mode = "jitter" if (mode_hash % 2 == 0) else "perlin"

		key = {
			"v": CACHE_VERSION,
			"kind": "noisy_sketch_mask",
			"shape_id": shape_id,
			"view_idx": int(view_idx),
			"noise_level": float(noise_level),
			"noise_mode": noise_mode,
			"sig": sig,
		}
		key_hash = _json_hash(key)
		path = self._path("noisy_sketch", shape_id, view_idx, key_hash)

		if self.enabled:
			cached = self._load_npz(path)
			if cached is not None:
				return cached["noisy_sketch_mask"]

		# Render noisy mask
		prev = (args.apply_jitter, args.apply_perlin, args.jitter_strength, args.perlin_strength, args.enable_partial_depth)
		args.apply_jitter = (noise_mode == "jitter")
		args.apply_perlin = (noise_mode == "perlin")
		args.jitter_strength = float(noise_level) if args.apply_jitter else 0.0
		args.perlin_strength = float(noise_level) if args.apply_perlin else 0.0
		args.perlin_scale = 2.0
		args.enable_partial_depth = False

		# Deterministic seed based on key
		_seed_everything(int(key_hash[:8], 16))

		TopoMapper = _load_topomapper_class()
		topo = TopoMapper(shape, args)
		noisy_data = topo.generate_pairs_to_memory(
			args.width, args.height, line_diameter=args.line_diameter, render_as_tube=True
		)

		args.apply_jitter, args.apply_perlin, args.jitter_strength, args.perlin_strength, args.enable_partial_depth = prev

		if noisy_data is None:
			return None

		noisy_sketch_mask = (noisy_data["edge_buffer"] > 0).astype(np.uint8)

		if self.enabled:
			to_save = {
				"noisy_sketch_mask": noisy_sketch_mask,
				"meta": _pack_object(key),
			}
			self._save_npz(path, to_save)

		return noisy_sketch_mask

	def get_legacy_noisy_render(
		self,
		shape,
		shape_id: str,
		view_idx: int,
		args,
		noise_level: float,
		noise_mode: Optional[str] = None,
	) -> Optional[Dict[str, Any]]:
		sig = _render_signature(args)

		if noise_mode is None:
			mode_key = {"shape_id": shape_id, "view_idx": int(view_idx), "noise_level": float(noise_level)}
			mode_hash = int(_json_hash(mode_key), 16)
			noise_mode = "jitter" if (mode_hash % 2 == 0) else "perlin"

		key = {
			"kind": "legacy_noisy",
			"v": CACHE_VERSION,
			"shape_id": shape_id,
			"view_idx": int(view_idx),
			"noise_level": float(noise_level),
			"noise_mode": noise_mode,
			"sig": sig,
		}
		key_hash = _json_hash(key)
		path = self._path("legacy_noisy", shape_id, view_idx, key_hash)

		if self.enabled:
			cached = self._load_npz(path)
			if cached is not None:
				return {
					"depth_buffer": cached["depth_buffer"],
					"edge_buffer": cached["edge_buffer"],
					"adjacency_list": _unpack_object(cached["adjacency_list"]),
					"coords": _unpack_object(cached["coords"]),
					"depth_fragments": _unpack_object(cached["depth_fragments"]),
					"edge_fragments": _unpack_object(cached["edge_fragments"]),
				}

		prev = (args.apply_jitter, args.apply_perlin, args.jitter_strength, args.perlin_strength, args.enable_partial_depth)
		args.apply_jitter = (noise_mode == "jitter")
		args.apply_perlin = (noise_mode == "perlin")
		args.jitter_strength = float(noise_level) if args.apply_jitter else 0.0
		args.perlin_strength = float(noise_level) if args.apply_perlin else 0.0
		args.perlin_scale = 2.0
		args.enable_partial_depth = False

		_seed_everything(int(key_hash[:8], 16))

		TopoMapper = _load_topomapper_class()
		topo = TopoMapper(shape, args)
		render_data = topo.generate_pairs_to_memory(
			args.width, args.height, line_diameter=args.line_diameter, render_as_tube=True
		)

		args.apply_jitter, args.apply_perlin, args.jitter_strength, args.perlin_strength, args.enable_partial_depth = prev

		if render_data is None:
			return None

		if self.enabled:
			to_save = {
				"depth_buffer": render_data["depth_buffer"],
				"edge_buffer": render_data["edge_buffer"],
				"adjacency_list": _pack_object(render_data["adjacency_list"]),
				"coords": _pack_object(render_data.get("coords", [])),
				"depth_fragments": _pack_object(render_data.get("depth_fragments", [])),
				"edge_fragments": _pack_object(render_data.get("edge_fragments", [])),
				"meta": _pack_object(key),
			}
			self._save_npz(path, to_save)

		return render_data
