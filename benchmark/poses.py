import numpy as np
from typing import List


def sample_random_camera_pose(
    seed: int,
    radius: float = float(3.0 * np.sqrt(3.0)),
) -> np.ndarray:
    """Sample a random camera pose (c2w) looking at the origin.

    This is intended as a fallback when fixed poses produce empty/invalid renders
    for certain shapes. The sampling is deterministic for a given seed.
    """

    def create_camera_matrix(position, look_at_direction):
        forward = np.array(look_at_direction, dtype=np.float32)
        forward = forward / (np.linalg.norm(forward) + 1e-8)
        up = np.array([0, 1, 0], dtype=np.float32)
        if abs(float(np.dot(forward, up))) > 0.98:
            up = np.array([0, 0, 1], dtype=np.float32)
        right = np.cross(up, forward)
        right = right / (np.linalg.norm(right) + 1e-8)
        up = np.cross(forward, right)

        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = forward
        c2w[:3, 3] = np.array(position, dtype=np.float32)
        return c2w

    rng = np.random.RandomState(int(seed) & 0xFFFFFFFF)
    for _ in range(64):
        v = rng.normal(size=(3,)).astype(np.float32)
        n = float(np.linalg.norm(v))
        if n < 1e-6:
            continue
        v = v / n
        pos = (v * float(radius)).tolist()
        direction = (-v).tolist()
        # Avoid near-pole directions that can make the camera basis unstable.
        if abs(float(direction[1])) > 0.98:
            continue
        return create_camera_matrix(pos, direction)

    # Very unlikely fallback.
    return create_camera_matrix([radius / np.sqrt(3), radius / np.sqrt(3), radius / np.sqrt(3)], [-1, -1, -1])


def get_fixed_camera_poses() -> List[np.ndarray]:
    """
    Returns 14 fixed camera poses: 6 orthographic + 8 isometric views.
    Each pose is a 4×4 camera-to-world transformation matrix.
    """
    def create_camera_matrix(position, look_at_direction):
        forward = np.array(look_at_direction, dtype=np.float32)
        forward = forward / (np.linalg.norm(forward) + 1e-8)
        up = np.array([0, 1, 0], dtype=np.float32)
        right = np.cross(up, forward)
        right = right / (np.linalg.norm(right) + 1e-8)
        up = np.cross(forward, right)

        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = forward
        c2w[:3, 3] = position
        return c2w

    poses = []

    # 6 Orthographic views (axis-aligned)
    distance = 3.0
    orthographic_views = [
        ([distance, 0, 0], [-1, 0, 0]),  # Right
        ([-distance, 0, 0], [1, 0, 0]),  # Left
        ([0, distance, 0], [0, -1, 0]),  # Top
        ([0, -distance, 0], [0, 1, 0]),  # Bottom
        ([0, 0, distance], [0, 0, -1]),  # Front
        ([0, 0, -distance], [0, 0, 1]),  # Back
    ]

    for pos, direction in orthographic_views:
        poses.append(create_camera_matrix(pos, direction))

    # 8 Isometric views (corners of a cube)
    iso_distance = distance * np.sqrt(3)  # diagonal distance
    for x in [1, -1]:
        for y in [1, -1]:
            for z in [1, -1]:
                pos = [x * iso_distance / np.sqrt(3), y * iso_distance / np.sqrt(3), z * iso_distance / np.sqrt(3)]
                direction = [-x, -y, -z]  # look towards origin
                poses.append(create_camera_matrix(pos, direction))

    return poses
