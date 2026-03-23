"""Generate ABC render dataset (.npz) using TopoMapper."""

from __future__ import annotations

import argparse
import os
from functools import partial

import numpy as np
from scipy.spatial.transform import Rotation
from tqdm.contrib.concurrent import process_map

from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.gp import gp_Pnt, gp_Quaternion, gp_Trsf, gp_Vec

from benchmark.complexity import calculate_shape_complexity
from dataset_tools.geometry import get_boundingbox
from dataset_tools.projection_core import generate_random_camera_pos
from dataset_tools.read_step_file import read_step_file
from dataset_tools.topomapper import TopoMapper


def _list_shape_ids(step_dir: str) -> list[str]:
    ids = []
    for name in sorted(os.listdir(step_dir)):
        if name.lower().endswith(".step"):
            ids.append(os.path.splitext(name)[0])
    return ids


def _normalize_shape(shape):
    center, extent = get_boundingbox([shape])

    trans = gp_Trsf()
    trans.SetTranslation(-gp_Vec(*center))

    scale = gp_Trsf()
    scale.SetScale(gp_Pnt(0, 0, 0), 2 / np.linalg.norm(extent))

    random_rot = Rotation.random()
    q = random_rot.as_quat()
    occ_quaternion = gp_Quaternion(q[0], q[1], q[2], q[3])

    rotation = gp_Trsf()
    rotation.SetRotation(occ_quaternion)

    brep_trans = BRepBuilderAPI_Transform(shape, scale * rotation * trans)
    return brep_trans.Shape()


def render_shape(shape_id: str, args) -> None:
    step_path = os.path.join(args.root, "step", f"{shape_id}.step")

    try:
        shape, num_shapes = read_step_file(step_path, verbosity=False, filter_num_shape=args.filter_num_shapes)
    except Exception:
        print(f"[skip] failed to read {shape_id}")
        return

    if shape is None or num_shapes > args.filter_num_shapes:
        return

    complexity_score, edge_count = calculate_shape_complexity(shape)
    if edge_count > args.filter_num_edges:
        return

    shape = _normalize_shape(shape)

    for idx in range(args.num_angles):
        focus, cam_pose = generate_random_camera_pos(seed=args.seed + idx)
        args.pose = cam_pose
        if args.focus != 0:
            args.focus = focus

        topo = TopoMapper(shape, args)
        out_path = os.path.join(args.root, args.zip_dir, f"{shape_id}_{idx}.npz")
        topo.generate_pairs(
            out_path,
            args.width,
            args.height,
            line_diameter=args.line_diameter,
            render_as_tube=True,
        )

        # Append metadata in a second write pass for compatibility with existing format.
        data = np.load(out_path, allow_pickle=True)
        payload = {k: data[k] for k in data.files}
        payload["complexity_score"] = float(complexity_score)
        payload["edge_count"] = int(edge_count)
        np.savez_compressed(out_path, **payload)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate ABC render dataset")
    parser.add_argument("--root", type=str, default="abc", help="Dataset root containing step/ and output dirs")
    parser.add_argument("--zip_dir", type=str, default="zip_clean", help="Output dir under --root")
    parser.add_argument("--num_angles", type=int, default=32)
    parser.add_argument("--num_cores", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--line_diameter", type=float, default=0.02)
    parser.add_argument("--tol", type=float, default=0.01)

    parser.add_argument("--fov", type=float, default=30)
    parser.add_argument("--focus", type=float, default=0)
    parser.add_argument("--location", nargs=3, type=float, default=[0, 0, 1])
    parser.add_argument("--direction", nargs=3, type=float, default=[0, 0, -1])
    parser.add_argument("--pose", default=None)

    parser.add_argument("--filter_num_shapes", type=int, default=10)
    parser.add_argument("--filter_num_edges", type=int, default=256)

    parser.add_argument("--apply_jitter", action="store_true", default=False)
    parser.add_argument("--apply_perlin", action="store_true", default=False)
    parser.add_argument("--jitter_strength", type=float, default=0.0)
    parser.add_argument("--perlin_strength", type=float, default=0.0)
    parser.add_argument("--perlin_scale", type=float, default=1.0)
    parser.add_argument("--enable_partial_depth", action="store_true", default=False)

    return parser.parse_args()


def main():
    args = parse_args()

    step_dir = os.path.join(args.root, "step")
    out_dir = os.path.join(args.root, args.zip_dir)
    os.makedirs(out_dir, exist_ok=True)

    shape_ids = _list_shape_ids(step_dir)
    if not shape_ids:
        print(f"No STEP files found in {step_dir}")
        return

    worker = partial(render_shape, args=args)
    process_map(worker, shape_ids, max_workers=args.num_cores, chunksize=1)


if __name__ == "__main__":
    main()
