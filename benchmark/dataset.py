from __future__ import annotations

import os
from typing import List, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
import traceback
import hashlib

from benchmark.poses import get_fixed_camera_poses
from benchmark.poses import sample_random_camera_pose
from benchmark.conditions import (
    accidental_pixel_ratio_from_fragments,
    accidental_pixel_ratio_for_edge_set,
    create_partial_depth_fast,
    render_partial_depth_occlusion_aware,
    select_known_edges_bfs,
)
from benchmark.cache import RenderCache
from benchmark.complexity import calculate_shape_complexity, get_topology_counts
from dataset_tools.read_step_file import read_step_file
from dataset_tools.geometry import get_boundingbox
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.gp import gp_Pnt, gp_Trsf, gp_Vec


NEAR = 3.0
FAR = 7.0


def _render_shape_with_noise_legacy(*args, **kwargs):
    """Optional legacy no-cache rendering backend.

    Public v1.0 benchmark is expected to run with cache-backed rendering.
    """
    try:
        from benchmark_diffusion import render_shape_with_noise as _impl
    except ImportError as exc:
        raise RuntimeError(
            "Legacy no-cache rendering requires benchmark_diffusion.py, which is not "
            "included in the v1.0 public benchmark path. Use --use_cache (default) "
            "or provide a compatible renderer backend."
        ) from exc
    return _impl(*args, **kwargs)


class BenchmarkDatasetV2(Dataset):
    """Dataset for benchmarking with fixed views and conditions.

        Returns a dict with:
            - sketch: (3, H, W) = [sketch_mask, partial_disp_hint, hint_valid]
      - gt_norm_depth: (1, H, W) normalized disparity in [0,1]
      - condition metadata
      - accidental_pixel_ratio (measured, not injected)
      - complexity proxies
    """

    def __init__(
        self,
        shape_ids: List[str],
        step_dir: str,
        args,
        noise_levels: List[float],
        completion_ratios: List[float],
        use_imperfect_to_perfect: bool = True,
        partial_depth_mode: str = "fast",
        views_subset: Optional[List[int]] = None,
        use_cache: bool = True,
        cache_dir: Optional[str] = None,
        render_retry_random: int = 2,
        render_retry_seed: int = 0,
    ):
        self.shape_ids = shape_ids
        self.step_dir = step_dir
        self.args = args
        self.noise_levels = noise_levels
        self.completion_ratios = completion_ratios
        self.use_imperfect_to_perfect = use_imperfect_to_perfect
        self.partial_depth_mode = partial_depth_mode
        self.use_cache = use_cache
        self.render_retry_random = int(render_retry_random)
        self.render_retry_seed = int(render_retry_seed)
        self.cache = None
        if cache_dir is not None:
            self.cache = RenderCache(cache_dir, enabled=use_cache)

        all_poses = get_fixed_camera_poses()
        if views_subset is None:
            # Use 4 isometric corner views (indices 6-9) by default
            self.fixed_poses = [all_poses[i] for i in [6, 7, 8, 9]]
        else:
            self.fixed_poses = [all_poses[i] for i in views_subset]

        self.shape_cache = {}
        self.shape_complexity_cache = {}
        self._fallback_pose_by_shape_view = {}
        self.samples = []

        for shape_id in shape_ids:
            for view_idx in range(len(self.fixed_poses)):
                # Baseline: no noise, empty depth
                self.samples.append((shape_id, view_idx, 0.0, 0.0, "baseline"))

                # Noisy only
                for noise in noise_levels:
                    if noise > 0:
                        self.samples.append((shape_id, view_idx, noise, 0.0, f"noise_{noise}"))

                # Partial only
                for completion in completion_ratios:
                    if completion > 0:
                        self.samples.append((shape_id, view_idx, 0.0, completion, f"partial_{completion}"))

                # Combined noisy + partial
                for noise in noise_levels:
                    for completion in completion_ratios:
                        if noise > 0 and completion > 0:
                            self.samples.append((shape_id, view_idx, noise, completion, f"noisy_{noise}_partial_{completion}"))

    def _load_and_normalize_shape(self, shape_id: str):
        if shape_id in self.shape_cache:
            return self.shape_cache[shape_id]

        step_path = os.path.join(self.step_dir, f"{shape_id}.step")
        try:
            shape, _ = read_step_file(step_path, verbosity=False)
            if shape is None:
                raise ValueError(f"Failed to load {shape_id}")
        except Exception:
            return None

        # Normalize shape: center and scale to fit in unit cube
        center, extent = get_boundingbox([shape])
        trans = gp_Trsf()
        trans.SetTranslation(-gp_Vec(*center))
        scale = gp_Trsf()
        scale.SetScale(gp_Pnt(0, 0, 0), 2 / np.linalg.norm(extent))
        brep_trans = BRepBuilderAPI_Transform(shape, scale * trans)
        normalized_shape = brep_trans.Shape()

        # Cache shape-level complexity stats (computed once per shape).
        try:
            topo_counts = get_topology_counts(normalized_shape)
            curve_complexity_score, curve_edge_count = calculate_shape_complexity(normalized_shape)
            topo_counts.update(
                {
                    "curve_complexity_score": float(curve_complexity_score),
                    "curve_edge_count": int(curve_edge_count),
                }
            )
            self.shape_complexity_cache[shape_id] = topo_counts
        except Exception:
            # Keep benchmark robust even if OCC complexity extraction fails.
            self.shape_complexity_cache[shape_id] = {
                "topo_num_faces": -1,
                "topo_num_edges": -1,
                "topo_num_vertices": -1,
                "curve_complexity_score": float("nan"),
                "curve_edge_count": -1,
            }

        self.shape_cache[shape_id] = normalized_shape
        return normalized_shape

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        shape_id, view_idx, noise_level, completion_ratio, condition = self.samples[idx]

        # Load and normalize shape
        shape = self._load_and_normalize_shape(shape_id)
        if shape is None:
            return None

        def _set_pose(pose_matrix: np.ndarray) -> None:
            self.args.pose = pose_matrix
            self.args.location = pose_matrix[:3, 3].tolist()
            self.args.direction = (-pose_matrix[:3, 2]).tolist()

        def _seed_for_attempt(attempt: int) -> int:
            payload = f"{self.render_retry_seed}|{shape_id}|{view_idx}|{attempt}".encode("utf-8")
            return int(hashlib.sha1(payload).hexdigest()[:8], 16)

        # Determine pose candidates: fixed pose first, then deterministic random retries.
        fallback_key = (shape_id, view_idx)
        pose_candidates = []
        pose_source_candidates = []
        if fallback_key in self._fallback_pose_by_shape_view:
            pose_candidates.append(self._fallback_pose_by_shape_view[fallback_key])
            pose_source_candidates.append("fallback_random")
        else:
            pose_candidates.append(self.fixed_poses[view_idx])
            pose_source_candidates.append("fixed")
            for attempt in range(1, max(0, self.render_retry_random) + 1):
                pose_candidates.append(sample_random_camera_pose(_seed_for_attempt(attempt)))
                pose_source_candidates.append(f"random_retry_{attempt}")

        # Render (with cache). If the fixed pose fails for a shape, retry with random poses.
        render_data = None
        pose_used = None
        pose_source_used = None
        last_exc = None

        for pose_matrix, pose_source in zip(pose_candidates, pose_source_candidates):
            _set_pose(pose_matrix)
            try:
                if self.cache is None:
                    # Fallback to on-the-fly rendering without cache
                    render_data = _render_shape_with_noise_legacy(
                        shape,
                        self.args,
                        noise_level,
                        completion_ratio,
                        use_imperfect_to_perfect=self.use_imperfect_to_perfect,
                    )
                else:
                    if self.use_imperfect_to_perfect:
                        # Always use clean render for targets/partials
                        clean_data = self.cache.get_clean_render(shape, shape_id, view_idx, self.args)
                        if clean_data is None:
                            render_data = None
                        else:
                            if noise_level > 0:
                                noisy_sketch_mask = self.cache.get_noisy_sketch_mask(
                                    shape, shape_id, view_idx, self.args, noise_level
                                )
                                if noisy_sketch_mask is None:
                                    render_data = None
                                else:
                                    render_data = {
                                        "noisy_sketch_mask": noisy_sketch_mask,
                                        "depth_buffer": clean_data["depth_buffer"],
                                        "edge_buffer": clean_data["edge_buffer"],
                                        "adjacency_list": clean_data["adjacency_list"],
                                        "perfect_edges": clean_data.get("perfect_edges", []),
                                        "coords": clean_data.get("coords", []),
                                        "depth_fragments": clean_data.get("depth_fragments", []),
                                        "edge_fragments": clean_data.get("edge_fragments", []),
                                    }
                            else:
                                render_data = {
                                    "noisy_sketch_mask": None,
                                    "depth_buffer": clean_data["depth_buffer"],
                                    "edge_buffer": clean_data["edge_buffer"],
                                    "adjacency_list": clean_data["adjacency_list"],
                                    "perfect_edges": clean_data.get("perfect_edges", []),
                                    "coords": clean_data.get("coords", []),
                                    "depth_fragments": clean_data.get("depth_fragments", []),
                                    "edge_fragments": clean_data.get("edge_fragments", []),
                                }
                    else:
                        # Legacy mode: cache full noisy render per noise_level
                        if noise_level > 0:
                            render_data = self.cache.get_legacy_noisy_render(
                                shape, shape_id, view_idx, self.args, noise_level
                            )
                        else:
                            render_data = self.cache.get_clean_render(shape, shape_id, view_idx, self.args)
            except Exception as e:
                last_exc = e
                render_data = None

            if render_data is not None:
                pose_used = pose_matrix
                pose_source_used = pose_source
                if pose_source != "fixed":
                    # Remember fallback pose so all conditions for this (shape, view) stay consistent.
                    self._fallback_pose_by_shape_view[fallback_key] = pose_matrix
                break

        if render_data is None:
            pid = os.getpid()
            print(
                f"[render-error pid={pid}] shape_id={shape_id} view_idx={view_idx} "
                f"noise={noise_level} completion={completion_ratio} retries={self.render_retry_random}"
            )
            if last_exc is not None:
                traceback.print_exc()
            return None

        # Extract render artifacts
        depth_buffer = render_data["depth_buffer"]
        edge_buffer = render_data["edge_buffer"]
        adjacency_list = render_data["adjacency_list"]
        coords = render_data.get("coords", [])
        depth_fragments = render_data.get("depth_fragments", [])
        edge_fragments = render_data.get("edge_fragments", [])

        # Per-edge curve complexity costs aligned with TopoMapper edge indices (if available)
        edge_curve_costs = render_data.get("edge_curve_costs", None)

        # Compute accidental pixel ratio (all edges):
        # proportion of primary-visible FG pixels that are overlap pixels.
        accidental_ratio_all, visible_fg_all, hidden_fg_all = accidental_pixel_ratio_from_fragments(
            edge_buffer, coords, depth_fragments, include_primary=True
        )

        # Convert depth to tensors
        depth_tensor = torch.from_numpy(depth_buffer.astype(np.float32))
        edge_buffer_tensor = torch.from_numpy(edge_buffer.astype(np.int64))

        # Ground truth: normalized disparity in [0,1]
        gt_mask = depth_tensor < 10.0 - 1e-6
        gt_disparity = torch.zeros_like(depth_tensor)
        disparity_near = 1.0 / NEAR
        disparity_far = 1.0 / FAR
        raw_disparity = 1.0 / (depth_tensor[gt_mask] + 1e-8)
        normalized_disparity = (raw_disparity - disparity_far) / (disparity_near - disparity_far + 1e-8)
        normalized_disparity = torch.clamp(normalized_disparity, 0.0, 1.0)
        gt_disparity[gt_mask] = normalized_disparity

        # Match training dataset semantics: mark all overlap/multi-fragment pixels invalid.
        occlusion_overlap = torch.zeros_like(gt_disparity, dtype=torch.bool)
        try:
            if coords is not None and len(coords) > 0:
                c = np.asarray(coords)
                if c.ndim == 2 and c.shape[1] == 2:
                    ys = torch.as_tensor(c[:, 0], dtype=torch.long)
                    xs = torch.as_tensor(c[:, 1], dtype=torch.long)
                    occlusion_overlap[ys, xs] = True
                else:
                    for y, x in coords:
                        occlusion_overlap[int(y), int(x)] = True
        except Exception:
            pass

        # Select known edges for partial depth (deterministic per shape/view/ratio)
        # Include all edge ids from adjacency list (keys + neighbors) and edge_buffer uniques
        all_edge_ids = set(int(k) for k in getattr(adjacency_list, "keys", lambda: [])())
        try:
            for nbrs in adjacency_list.values():
                for n in nbrs:
                    all_edge_ids.add(int(n))
        except Exception:
            pass
        # Always union with edge_buffer uniques to cover isolated edges
        try:
            all_edge_ids |= set(int(x) for x in np.unique(edge_buffer) if int(x) > 0)
        except Exception:
            pass

        # Ensure adjacency_list includes isolated edges so BFS can traverse them
        if all_edge_ids:
            try:
                adjacency_list = dict(adjacency_list)
                for e_id in all_edge_ids:
                    adjacency_list.setdefault(int(e_id), [])
            except Exception:
                pass

        if completion_ratio >= 1.0:
            known_edges = sorted(all_edge_ids)
        elif completion_ratio <= 0.0:
            known_edges = []
        else:
            seed_payload = f"{self.render_retry_seed}|{shape_id}|{view_idx}|{float(completion_ratio):.6f}".encode("utf-8")
            sel_seed = int(hashlib.sha1(seed_payload).hexdigest()[:8], 16)
            known_edges = select_known_edges_bfs(adjacency_list, completion_ratio, seed=sel_seed)

        known_edge_ids = set(int(e) for e in known_edges)
        unknown_edge_ids = set(int(e) for e in all_edge_ids if int(e) not in known_edge_ids)
        known_edge_tensor = torch.tensor(list(known_edge_ids), dtype=torch.int64)

        # Unknown/known pixel masks (computed later after depth_channel is built)
        known_pixel_mask = None
        unknown_pixel_mask = None
        eps = 1e-6

        # Build input channels (match training pipeline semantics)
        if noise_level > 0 and self.use_imperfect_to_perfect:
            noisy_sketch_mask = render_data.get("noisy_sketch_mask", None)
            if noisy_sketch_mask is None:
                return None
            rgb_channel = torch.from_numpy(noisy_sketch_mask.astype(np.float32))
        else:
            # Clean mode uses binary fg mask
            rgb_channel = gt_mask.float()

        if completion_ratio <= 0.0:
            depth_channel = torch.zeros_like(gt_disparity)
            valid_channel = torch.zeros_like(gt_disparity)
        elif completion_ratio >= 1.0:
            depth_channel = gt_disparity.clone()
            primary_known = torch.isin(edge_buffer_tensor, known_edge_tensor)
            valid_channel = (primary_known & (gt_disparity > eps) & (~occlusion_overlap)).float()
        else:
            if self.partial_depth_mode == "occlusion_aware":
                # Training uses occlusion-aware rendering of known edges
                depth_channel = render_partial_depth_occlusion_aware(
                    known_edges,
                    gt_disparity,
                    edge_buffer_tensor,
                    coords,
                    depth_fragments,
                    edge_fragments,
                )
            else:
                # Fast approximation: reveal disparity only where known edges are the primary edge
                render_mask = torch.isin(edge_buffer_tensor, known_edge_tensor)
                depth_channel = gt_disparity * render_mask.float()

            primary_known = torch.isin(edge_buffer_tensor, known_edge_tensor)
            valid_channel = (primary_known & (gt_disparity > eps) & (~occlusion_overlap)).float()

        # Option B semantics: keep rendered hint values in depth_channel.
        # Trustworthiness is encoded explicitly in valid_channel. Guarantee that all
        # valid==1 pixels are exact GT disparity values.
        depth_channel = torch.where(valid_channel > 0.5, gt_disparity, depth_channel)

        fg_pixel_mask = gt_disparity > eps

        # IMPORTANT: In training, the partial-depth channel is rendered occlusion-aware and
        # can contain depths for edges that are *behind* an unknown primary edge at a pixel.
        # Therefore, (depth_channel > 0) is NOT equivalent to "pixels where the answer is given".
        # For evaluation/analysis, we track primary-visible known/unknown masks separately.
        hinted_pixel_mask = depth_channel > eps
        known_pixel_mask = valid_channel > 0.5
        primary_known_pixel_mask = fg_pixel_mask & torch.isin(
            edge_buffer_tensor,
            known_edge_tensor,
        )
        primary_unknown_pixel_mask = fg_pixel_mask & (~primary_known_pixel_mask)
        # Pixels where a hint exists but the primary edge is unknown (hint is likely occluded/behind).
        hint_conflict_pixel_mask = hinted_pixel_mask & primary_unknown_pixel_mask

        # Back-compat: keep unknown_pixel_mask, but define it as "primary unknown" rather than
        # "not hinted" (the old definition was misleading for occlusion-aware hints).
        unknown_pixel_mask = primary_unknown_pixel_mask

        sketch_input = torch.stack([rgb_channel, depth_channel, valid_channel], dim=0)
        gt_norm_depth = gt_disparity.unsqueeze(0)

        # Complexity proxies (all vs unknown)
        num_edges_all = int(len(all_edge_ids))
        num_edges_unknown = int(len(unknown_edge_ids))
        stroke_pixels = int((rgb_channel > 0).sum().item())
        shape_complexity = self.shape_complexity_cache.get(shape_id, {})

        # Accidental pixel ratio for non-hint pixels (primary-unknown subset):
        # proportion of primary-unknown pixels that are overlap pixels.
        accidental_ratio_unknown, visible_fg_unknown, hidden_fg_unknown = accidental_pixel_ratio_for_edge_set(
            edge_buffer,
            coords,
            depth_fragments,
            edge_fragments,
            unknown_edge_ids,
            include_primary=True,
        )

        # Curve complexity score restricted to unknown edges (TopoMapper-aligned indices)
        curve_complexity_total = float("nan")
        curve_complexity_unknown = float("nan")
        curve_edge_count_total = -1
        curve_edge_count_unknown = -1
        if isinstance(edge_curve_costs, np.ndarray) and edge_curve_costs.ndim == 1:
            curve_edge_count_total = int((edge_curve_costs > 0).sum())
            curve_complexity_total = float(edge_curve_costs.sum())
            if unknown_edge_ids:
                idxs = [i for i in unknown_edge_ids if 0 <= int(i) < edge_curve_costs.shape[0]]
                curve_edge_count_unknown = int(len(idxs))
                if idxs:
                    curve_complexity_unknown = float(edge_curve_costs[np.array(idxs, dtype=np.int64)].sum())

        # Fallback for legacy caches / failures where per-edge costs are unavailable.
        if not np.isfinite(curve_complexity_total):
            total_score_shape = float(shape_complexity.get("curve_complexity_score", float("nan")))
            total_edges_shape = int(shape_complexity.get("curve_edge_count", -1))
            if np.isfinite(total_score_shape):
                curve_complexity_total = total_score_shape
            if curve_edge_count_total < 0 and total_edges_shape >= 0:
                curve_edge_count_total = total_edges_shape

            # Unknown subset fallback: proportional split by edge-count ratio.
            if (not np.isfinite(curve_complexity_unknown)) and np.isfinite(curve_complexity_total):
                curve_edge_count_unknown = int(len(unknown_edge_ids))
                if curve_edge_count_total > 0:
                    frac_unknown = min(1.0, max(0.0, curve_edge_count_unknown / float(curve_edge_count_total)))
                    curve_complexity_unknown = float(
                        curve_complexity_total * frac_unknown
                    )
                else:
                    curve_complexity_unknown = 0.0

        # Final guard: never emit NaN in benchmark outputs.
        if not np.isfinite(curve_complexity_total):
            curve_complexity_total = 0.0
        if not np.isfinite(curve_complexity_unknown):
            curve_complexity_unknown = 0.0

        return {
            "sketch": sketch_input,
            "gt_norm_depth": gt_norm_depth,
            "shape_id": shape_id,
            "view_idx": view_idx,
            "pose_source": pose_source_used or "fixed",
            "condition": condition,
            "noise_level": noise_level,
            "noise_scalar": float(max(0.0, min(1.0, float(noise_level) / 0.15))) if noise_level > 0 else 0.0,
            "completion_ratio": completion_ratio,
            "known_edge_count": int(len(known_edge_ids)),
            "unknown_edge_count": int(len(unknown_edge_ids)),
            "known_pixel_count": int(known_pixel_mask.sum().item()) if known_pixel_mask is not None else 0,
            "unknown_pixel_count": int(unknown_pixel_mask.sum().item()) if unknown_pixel_mask is not None else 0,
            "known_pixel_mask": known_pixel_mask,
            "unknown_pixel_mask": unknown_pixel_mask,
            "primary_known_pixel_count": int(primary_known_pixel_mask.sum().item()),
            "primary_unknown_pixel_count": int(primary_unknown_pixel_mask.sum().item()),
            "hint_conflict_pixel_count": int(hint_conflict_pixel_mask.sum().item()),
            "primary_known_pixel_mask": primary_known_pixel_mask,
            "primary_unknown_pixel_mask": primary_unknown_pixel_mask,
            "hint_conflict_pixel_mask": hint_conflict_pixel_mask,
            "accidental_pixel_ratio": float(accidental_ratio_unknown if completion_ratio > 0 else accidental_ratio_all),
            "accidental_pixel_ratio_all": float(accidental_ratio_all),
            "accidental_pixel_ratio_unknown": float(accidental_ratio_unknown),
            "visible_fg_pixels": int(visible_fg_unknown if completion_ratio > 0 else visible_fg_all),
            "overlap_pixel_count": int(hidden_fg_unknown if completion_ratio > 0 else hidden_fg_all),
            "hidden_fg_fragments": int(hidden_fg_unknown if completion_ratio > 0 else hidden_fg_all),
            "visible_fg_pixels_all": int(visible_fg_all),
            "overlap_pixel_count_all": int(hidden_fg_all),
            "hidden_fg_fragments_all": int(hidden_fg_all),
            "visible_fg_pixels_unknown": int(visible_fg_unknown),
            "overlap_pixel_count_unknown": int(hidden_fg_unknown),
            "hidden_fg_fragments_unknown": int(hidden_fg_unknown),
            "num_edges": num_edges_unknown if completion_ratio > 0 else num_edges_all,
            "num_edges_all": num_edges_all,
            "num_edges_unknown": num_edges_unknown,
            "stroke_pixels": stroke_pixels,
            "topo_num_faces": int(shape_complexity.get("topo_num_faces", -1)),
            "topo_num_edges": int(shape_complexity.get("topo_num_edges", -1)),
            "topo_num_vertices": int(shape_complexity.get("topo_num_vertices", -1)),
            "curve_complexity_score": float(curve_complexity_unknown if completion_ratio > 0 else curve_complexity_total),
            "curve_edge_count": int(curve_edge_count_unknown if completion_ratio > 0 else curve_edge_count_total),
            "curve_complexity_score_all": float(curve_complexity_total),
            "curve_edge_count_all": int(curve_edge_count_total),
            "curve_complexity_score_unknown": float(curve_complexity_unknown),
            "curve_edge_count_unknown": int(curve_edge_count_unknown),
        }
