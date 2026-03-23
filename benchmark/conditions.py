from __future__ import annotations

from typing import Iterable, Tuple
import numpy as np
import torch
import hashlib
from typing import Dict, List, Optional, Set


def accidental_pixel_ratio_from_fragments(
    edge_buffer: np.ndarray,
    coords: Iterable[Tuple[int, int]],
    depth_fragments: Iterable,
    include_primary: bool = True,
) -> Tuple[float, int, int]:
    """
    Compute accidental pixel ratio from fragment stacks.

    Notes:
        TopoMapper returns (coords, depth_fragments, edge_fragments) for pixels
        where multiple depth intersections exist. Each coord is one pixel with
        overlap. We count overlap pixels once each (not per hidden fragment).

    Args:
        edge_buffer: primary-visible edge id image
        coords: iterable of (y, x)
        depth_fragments: list of per-pixel hidden-fragment depth lists
        include_primary: if True, count one visible fragment per pixel

    Returns:
        ratio, visible_fg_pixels, overlap_pixels
    """
    if include_primary:
        visible = int(np.count_nonzero(np.asarray(edge_buffer) > 0))
    else:
        visible = 0

    overlap = 0
    seen_coords = set()
    for i, coord in enumerate(coords):
        try:
            y, x = int(coord[0]), int(coord[1])
        except Exception:
            continue
        key = (y, x)
        if key in seen_coords:
            continue

        try:
            ds = depth_fragments[i]
            has_hidden = len(ds) > 0
        except Exception:
            has_hidden = False

        if has_hidden:
            overlap += 1
            seen_coords.add(key)

    # Pixel-level proportion of visible FG pixels that have overlap.
    ratio = float(overlap / visible) if visible > 0 else 0.0
    return ratio, visible, overlap


def accidental_pixel_ratio_for_edge_set(
    edge_buffer: np.ndarray,
    coords: Iterable[Tuple[int, int]],
    depth_fragments: Iterable,
    edge_fragments: Iterable,
    edge_ids: Set[int],
    include_primary: bool = True,
) -> Tuple[float, int, int]:
    """Pixel-overlap accidental ratio restricted to a subset of primary edges.

    Definition for a subset (e.g., unknown/non-hint pixels):
        - visible: pixels whose *primary* edge id is in edge_ids
        - overlap: among those visible pixels, count unique coords with >=1 hidden fragment

    This matches: proportion of non-hint pixels that contain overlapping depth values
    (2+ edges at pixel with at least one occluded behind the primary).
    """

    if include_primary:
        visible = int(np.count_nonzero(np.isin(edge_buffer, list(edge_ids))))
    else:
        visible = 0

    overlap = 0
    edge_ids_int = set(int(e) for e in edge_ids)
    seen_coords = set()

    for i, (y, x) in enumerate(coords):
        try:
            key = (int(y), int(x))
        except Exception:
            continue
        if key in seen_coords:
            continue

        # Restrict to pixels whose primary-visible edge is in the subset.
        try:
            if int(edge_buffer[int(y), int(x)]) not in edge_ids_int:
                continue
        except Exception:
            continue

        # coords from TopoMapper already represent overlap pixels (len>1 intersections),
        # but keep a defensive hidden-fragment check.
        frag_depths = depth_fragments[i]
        try:
            has_hidden = len(frag_depths) > 0
            if has_hidden:
                overlap += 1
                seen_coords.add(key)
        except Exception:
            continue

    ratio = float(overlap / visible) if visible > 0 else 0.0
    return ratio, visible, overlap


def select_known_edges_bfs(
    adjacency_list: Dict[int, List[int]],
    completion_ratio: float,
    seed: int,
) -> List[int]:
    """Deterministically select a connected prefix of edges via BFS traversal.

    Mirrors the training dataset behavior (BFS traversal with shuffled neighbors),
    but makes it deterministic for benchmarking/caching.
    """
    from collections import deque
    import random

    if completion_ratio <= 0:
        return []

    rng = random.Random(int(seed) & 0xFFFFFFFF)

    visited = set()
    all_paths: List[List[int]] = []

    all_nodes = list(adjacency_list.keys())
    rng.shuffle(all_nodes)

    # Ensure all nodes appear even if only in neighbor lists
    for node_list in adjacency_list.values():
        for n in node_list:
            if n not in adjacency_list and n not in all_nodes:
                all_nodes.append(n)
    rng.shuffle(all_nodes)

    for start_node in all_nodes:
        if start_node in visited:
            continue
        q = deque([start_node])
        visited.add(start_node)
        path: List[int] = []
        while q:
            node = q.popleft()
            path.append(int(node))
            neighbors = list(adjacency_list.get(int(node), []))
            rng.shuffle(neighbors)
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    q.append(neighbor)
        all_paths.append(path)

    rng.shuffle(all_paths)
    full_path: List[int] = [edge for component in all_paths for edge in component]
    if not full_path:
        return []

    num_edges_to_keep = max(1, int(len(full_path) * float(completion_ratio)))
    return list(full_path[:num_edges_to_keep])


def create_partial_depth_fast(
    depth_map: np.ndarray,
    edge_buffer: np.ndarray,
    adjacency_list: dict,
    completion_ratio: float,
) -> np.ndarray:
    """
    Create partial depth using BFS traversal over edges (fast approximation).
    Matches the logic from benchmark_diffusion.py.
    """
    from collections import deque
    import random

    visited = set()
    all_paths = []

    all_nodes = list(adjacency_list.keys())
    random.shuffle(all_nodes)

    for start_node in all_nodes:
        if start_node in visited:
            continue
        queue = deque([start_node])
        visited.add(start_node)
        path = []

        while queue:
            node = queue.popleft()
            path.append(node)
            neighbors = list(adjacency_list.get(node, []))
            random.shuffle(neighbors)
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        all_paths.append(path)

    random.shuffle(all_paths)
    full_path = [edge for component in all_paths for edge in component]

    num_edges_to_keep = max(1, int(len(full_path) * completion_ratio))
    edges_to_keep = set(full_path[:num_edges_to_keep])

    partial_depth = np.zeros_like(depth_map)
    mask = np.isin(edge_buffer, list(edges_to_keep))
    partial_depth[mask] = depth_map[mask]

    return partial_depth


def render_partial_depth_occlusion_aware(
    perfect_edges,
    primary_disparity: torch.Tensor,
    edge_buffer: torch.Tensor,
    coords,
    depth_fragments,
    edge_fragments,
) -> torch.Tensor:
    """
    Render partial depth using fragment stacks (occlusion-aware),
    matching dataset.py / benchmark_diffusion.py logic.
    """
    H, W = primary_disparity.shape
    render_mask = torch.isin(edge_buffer, torch.tensor(perfect_edges, dtype=torch.int64))
    rendered_disparity = primary_disparity * render_mask.float()

    # Compute disparity normalization constants
    NEAR = 3.0
    FAR = 7.0
    disparity_near = 1.0 / NEAR
    disparity_far = 1.0 / FAR

    for i, (y, x) in enumerate(coords):
        frag_depths = depth_fragments[i]
        frag_edges = edge_fragments[i]
        relevant_depths = [d for d, e_id in zip(frag_depths, frag_edges) if e_id in perfect_edges]
        if not relevant_depths:
            continue

        all_possible_depths = relevant_depths
        if edge_buffer[y, x] in perfect_edges:
            norm_disp = primary_disparity[y, x].item()
            raw_disp = norm_disp * (disparity_near - disparity_far) + disparity_far
            depth = 1.0 / (raw_disp + 1e-8)
            all_possible_depths.append(depth)

        closest_depth = min(d for d in all_possible_depths if d > 1e-6)
        raw_disparity = 1.0 / (closest_depth + 1e-8)
        normalized_disparity = (raw_disparity - disparity_far) / (disparity_near - disparity_far + 1e-8)
        normalized_disparity = float(max(0.0, min(1.0, normalized_disparity)))
        rendered_disparity[y, x] = normalized_disparity

    return rendered_disparity
