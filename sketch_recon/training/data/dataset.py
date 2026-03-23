import glob
import os
import random
from collections import deque

import numpy as np
import torch
from torch.utils.data import Dataset

NEAR = 3.0
FAR = 7.0


class OcclusionAwareSequentialDataset(Dataset):
    """Dataset that loads clean or noisy renders and builds training tensors."""

    def __init__(
        self,
        zip_dir,
        indices,
        transforms=None,
        use_noisy=False,
        zip_noisy_dir=None,
        noisy_shapes=None,
        occlusion_aware_partial_depth: bool = True,
    ):
        self.use_noisy = use_noisy
        self.occlusion_aware_partial_depth = bool(occlusion_aware_partial_depth)

        if use_noisy:
            if zip_noisy_dir is None:
                raise ValueError("zip_noisy_dir must be provided when use_noisy=True")
            self.zip_paths = sorted(glob.glob(os.path.join(zip_noisy_dir, "*.npz")))
            print(f"Loading EXCLUSIVELY from noisy dataset: {len(self.zip_paths)} files in {zip_noisy_dir}")
        else:
            self.zip_paths = sorted(glob.glob(os.path.join(zip_dir, "*.npz")))
            print(f"Loading EXCLUSIVELY from clean dataset: {len(self.zip_paths)} files in {zip_dir}")

        self.indices = indices

        if max(indices) >= len(self.zip_paths):
            raise ValueError(f"Index {max(indices)} out of range for dataset with {len(self.zip_paths)} files")

    def __len__(self):
        return len(self.indices)

    def _bfs_traversal(self, adj, start_node, visited):
        q = deque([start_node])
        visited.add(start_node)
        path = []
        while q:
            node = q.popleft()
            path.append(node)

            neighbors = list(adj.get(node, []))
            random.shuffle(neighbors)

            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    q.append(neighbor)
        return path

    def _get_all_traversal_paths(self, adj):
        all_component_paths = []
        nodes_in_a_path = set()
        all_nodes = list(adj.keys())

        for node_list in adj.values():
            for node in node_list:
                if node not in all_nodes:
                    all_nodes.append(node)

        random.shuffle(all_nodes)

        for node in all_nodes:
            if node in nodes_in_a_path:
                continue

            new_path = self._bfs_traversal(adj, node, nodes_in_a_path)
            all_component_paths.append(new_path)

        random.shuffle(all_component_paths)
        full_path = [edge for component in all_component_paths for edge in component]

        return full_path

    def _render_map_primary_only(self, edges_to_render, primary_disparity, edge_buffer):
        render_mask = torch.isin(edge_buffer, torch.tensor(edges_to_render, dtype=torch.int64))
        return primary_disparity * render_mask.float()

    def _render_map_occlusion_aware(
        self,
        edges_to_render,
        primary_disparity,
        edge_buffer,
        coords,
        depth_fragments,
        edge_fragments,
    ):
        rendered_disparity = primary_disparity * torch.isin(
            edge_buffer, torch.tensor(edges_to_render, dtype=torch.int64)
        ).float()

        disparity_near = 1.0 / NEAR
        disparity_far = 1.0 / FAR

        for i, (y, x) in enumerate(coords):
            frag_depths = depth_fragments[i]
            frag_edges = edge_fragments[i]

            relevant_depths = [d for d, e_id in zip(frag_depths, frag_edges) if e_id in edges_to_render]
            if not relevant_depths:
                continue

            all_possible_depths = relevant_depths
            if edge_buffer[y, x] in edges_to_render:
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

    def __getitem__(self, idx):
        shape_idx = self.indices[idx]
        loaded_data = np.load(self.zip_paths[shape_idx], allow_pickle=True)

        depth_buffer = torch.from_numpy(loaded_data["depth_buffer"].astype(np.float32))
        edge_buffer = torch.from_numpy(loaded_data["edge_buffer"].astype(np.int64))
        coords = loaded_data["coords"]
        depth_fragments = loaded_data["depth_fragments"]
        edge_fragments = loaded_data["edge_fragments"]
        adj_list_items = loaded_data["adjacency_list"]
        adj_list = {int(k): set(v) for k, v in adj_list_items}

        mask = depth_buffer < 10.0 - 1e-6
        disparity_img = torch.zeros_like(depth_buffer)

        raw_disparity = 1.0 / (depth_buffer[mask] + 1e-8)
        disparity_near = 1.0 / NEAR
        disparity_far = 1.0 / FAR
        normalized_disparity = (raw_disparity - disparity_far) / (disparity_near - disparity_far + 1e-8)
        normalized_disparity = torch.clamp(normalized_disparity, 0.0, 1.0)
        disparity_img[mask] = normalized_disparity

        occlusion_overlap = torch.zeros_like(disparity_img, dtype=torch.bool)
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

        if "noisy_sketch_mask" in loaded_data:
            rgb_tensor = torch.from_numpy(loaded_data["noisy_sketch_mask"].astype(np.float32))
        else:
            rgb_tensor = mask.float()

        if "noise_scalar" in loaded_data:
            raw_noise = float(loaded_data["noise_scalar"])
            if raw_noise > 0 and raw_noise < 0.1:
                old_noise_min = 0.02
                old_noise_max = 0.08
                noise_scalar = (raw_noise - old_noise_min) / (old_noise_max - old_noise_min)
                noise_scalar = max(0.0, min(1.0, noise_scalar))
            else:
                noise_scalar = max(0.0, min(1.0, raw_noise))
        else:
            noise_scalar = 0.0

        has_perfect_edges = self.use_noisy and ("perfect_edges" in loaded_data)

        if has_perfect_edges:
            perfect_edges = loaded_data["perfect_edges"]
            if len(perfect_edges) == 0:
                hint = torch.full_like(rgb_tensor, 0)
                valid = torch.full_like(rgb_tensor, 0)
                model_input_tensor = torch.stack([rgb_tensor, hint, valid], dim=0)
            else:
                edges_to_render = perfect_edges.tolist()
                if self.occlusion_aware_partial_depth:
                    partial_depth_channel = self._render_map_occlusion_aware(
                        edges_to_render,
                        disparity_img,
                        edge_buffer,
                        coords,
                        depth_fragments,
                        edge_fragments,
                    )
                else:
                    partial_depth_channel = self._render_map_primary_only(edges_to_render, disparity_img, edge_buffer)

                primary_known = torch.isin(edge_buffer, torch.tensor(edges_to_render, dtype=torch.int64))
                valid = (primary_known & (disparity_img > 1e-6) & (~occlusion_overlap)).float()
                partial_depth_channel = torch.where(valid > 0.5, disparity_img, partial_depth_channel)
                partial_depth_channel[partial_depth_channel == 0] = 0
                model_input_tensor = torch.stack([rgb_tensor, partial_depth_channel, valid], dim=0)
        else:
            full_path = self._get_all_traversal_paths(adj_list)
            if not full_path:
                h, w = 256, 256
                return torch.zeros((3, h, w)), torch.zeros((1, h, w)), noise_scalar

            if random.random() < 0.5:
                hint = torch.full_like(rgb_tensor, 0)
                valid = torch.full_like(rgb_tensor, 0)
                model_input_tensor = torch.stack([rgb_tensor, hint, valid], dim=0)
            else:
                min_step = int((len(full_path) - 1) * 0.1)
                max_step = int((len(full_path) - 1) * 0.9)
                step = random.randint(min_step, max_step)
                edges_to_render = full_path[:step]

                if self.occlusion_aware_partial_depth:
                    partial_depth_channel = self._render_map_occlusion_aware(
                        edges_to_render,
                        disparity_img,
                        edge_buffer,
                        coords,
                        depth_fragments,
                        edge_fragments,
                    )
                else:
                    partial_depth_channel = self._render_map_primary_only(edges_to_render, disparity_img, edge_buffer)

                primary_known = torch.isin(edge_buffer, torch.tensor(edges_to_render, dtype=torch.int64))
                valid = (primary_known & (disparity_img > 1e-6) & (~occlusion_overlap)).float()
                partial_depth_channel = torch.where(valid > 0.5, disparity_img, partial_depth_channel)
                partial_depth_channel[partial_depth_channel == 0] = 0
                model_input_tensor = torch.stack([rgb_tensor, partial_depth_channel, valid], dim=0)

        return model_input_tensor, disparity_img.unsqueeze(dim=0), noise_scalar
