import glob
import os
from typing import List
from sklearn.utils import shuffle


def get_test_split(zip_dir: str, val_size: float = 0.05, test_size: float = 0.05) -> List[str]:
    """
    Get test split using the same logic as full_train_diffusion.py.

    Args:
        zip_dir: Directory with .npz files (use abc/zip_clean for consistency)
        val_size: Validation split ratio
        test_size: Test split ratio

    Returns:
        List of test shape IDs (8-character prefixes)
    """
    clean_files = sorted(glob.glob(os.path.join(zip_dir, "*.npz")))
    clean_basenames = [os.path.basename(x) for x in clean_files]

    # Get all unique shapes from clean dataset
    all_shapes = sorted(list(set([x[:8] for x in clean_basenames])))

    # Split shapes into train/val/test (shape-level split)
    shapes = shuffle(all_shapes, random_state=0)
    num_shapes = len(shapes)

    train_size = int(num_shapes * (1 - (val_size + test_size)))
    val_size_actual = int(num_shapes * val_size)

    # Test shapes are everything after train + val
    test_shapes = shapes[train_size + val_size_actual:]
    return list(test_shapes)


def load_shape_ids(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]
    return [ln for ln in lines if ln]


def save_shape_ids(path: str, ids: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(ids))
        f.write("\n")
