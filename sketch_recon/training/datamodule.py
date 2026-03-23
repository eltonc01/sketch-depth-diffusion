"""Lightning DataModule for sketch-depth diffusion training."""

import glob
import os

import pytorch_lightning as pl
from sklearn.utils import shuffle
from torch.utils.data import DataLoader

from sketch_recon.training.data.dataset import OcclusionAwareSequentialDataset


class WireframeDataModule(pl.LightningDataModule):
	def __init__(
		self,
		reg_dir,
		depth_dir,
		batch_size=64,
		num_workers=4,
		val_size=0.05,
		test_size=0.05,
		threshold=None,
		use_noisy_data=False,
		occlusion_aware_partial_depth: bool = True,
	):
		super().__init__()
		self.reg_dir = reg_dir
		self.depth_dir = depth_dir
		self.batch_size = batch_size
		self.num_workers = num_workers
		self.val_size = val_size
		self.test_size = test_size
		self.threshold = threshold
		self.use_noisy_data = use_noisy_data
		self.occlusion_aware_partial_depth = bool(occlusion_aware_partial_depth)
		self.transform = None

	def setup(self, stage=None):
		if self.use_noisy_data:
			print("\n" + "=" * 70)
			print("DATA MODE: NOISY (imperfect_to_perfect)")
			print("  - Input: Noisy sketch masks from abc/zip_noisy")
			print("  - Target: Clean depth from abc/zip_noisy (stored alongside noisy sketch)")
			print("=" * 70 + "\n")

			noisy_files = sorted(glob.glob(os.path.join("abc/zip_noisy", "*.npz")))
			noisy_basenames = [os.path.basename(x) for x in noisy_files]
			all_shapes = sorted(list(set([x[:8] for x in noisy_basenames])))

			shapes = shuffle(all_shapes, random_state=0)
			num_shapes = len(shapes)
			train_size = int(num_shapes * (1 - (self.val_size + self.test_size)))
			val_size = int(num_shapes * self.val_size)

			train_shapes = set(shapes[:train_size])
			val_shapes = set(shapes[train_size : train_size + val_size])

			train_indices = [i for i, f in enumerate(noisy_basenames) if f[:8] in train_shapes]
			val_indices = [i for i, f in enumerate(noisy_basenames) if f[:8] in val_shapes]

			train_indices = shuffle(train_indices, random_state=0)
			val_indices = shuffle(val_indices, random_state=0)

			print(f"Dataset split: {len(train_indices)} train, {len(val_indices)} val samples")
			print(f"Shapes: {len(train_shapes)} train, {len(val_shapes)} val (all noisy)")

			self.train_ds = OcclusionAwareSequentialDataset(
				"abc/zip_clean",
				train_indices,
				self.transform,
				use_noisy=True,
				zip_noisy_dir="abc/zip_noisy",
				occlusion_aware_partial_depth=self.occlusion_aware_partial_depth,
			)
			self.val_ds = OcclusionAwareSequentialDataset(
				"abc/zip_clean",
				val_indices,
				self.transform,
				use_noisy=True,
				zip_noisy_dir="abc/zip_noisy",
				occlusion_aware_partial_depth=self.occlusion_aware_partial_depth,
			)
		else:
			print("\n" + "=" * 70)
			print("DATA MODE: CLEAN (perfect_to_perfect)")
			print("  - Input: Clean sketch masks from abc/zip_clean")
			print("  - Target: Clean depth from abc/zip_clean")
			print("=" * 70 + "\n")

			clean_files = sorted(glob.glob(os.path.join("abc/zip_clean", "*.npz")))
			clean_basenames = [os.path.basename(x) for x in clean_files]
			all_shapes = sorted(list(set([x[:8] for x in clean_basenames])))

			shapes = shuffle(all_shapes, random_state=0)
			num_shapes = len(shapes)
			train_size = int(num_shapes * (1 - (self.val_size + self.test_size)))
			val_size = int(num_shapes * self.val_size)

			train_shapes = set(shapes[:train_size])
			val_shapes = set(shapes[train_size : train_size + val_size])

			train_indices = [i for i, f in enumerate(clean_basenames) if f[:8] in train_shapes]
			val_indices = [i for i, f in enumerate(clean_basenames) if f[:8] in val_shapes]

			train_indices = shuffle(train_indices, random_state=0)
			val_indices = shuffle(val_indices, random_state=0)

			print(f"Dataset split: {len(train_indices)} train, {len(val_indices)} val samples")
			print(f"Shapes: {len(train_shapes)} train, {len(val_shapes)} val (all clean)")

			self.train_ds = OcclusionAwareSequentialDataset(
				"abc/zip_clean",
				train_indices,
				self.transform,
				use_noisy=False,
				occlusion_aware_partial_depth=self.occlusion_aware_partial_depth,
			)
			self.val_ds = OcclusionAwareSequentialDataset(
				"abc/zip_clean",
				val_indices,
				self.transform,
				use_noisy=False,
				occlusion_aware_partial_depth=self.occlusion_aware_partial_depth,
			)

	def train_dataloader(self):
		dl_kwargs = {}
		if self.num_workers and self.num_workers > 0:
			dl_kwargs["persistent_workers"] = True
			dl_kwargs["prefetch_factor"] = 2
		return DataLoader(
			self.train_ds,
			batch_size=self.batch_size,
			shuffle=True,
			num_workers=self.num_workers,
			pin_memory=True,
			drop_last=True,
			**dl_kwargs,
		)

	def val_dataloader(self):
		dl_kwargs = {}
		if self.num_workers and self.num_workers > 0:
			dl_kwargs["persistent_workers"] = True
			dl_kwargs["prefetch_factor"] = 2
		return DataLoader(
			self.val_ds,
			batch_size=self.batch_size,
			shuffle=False,
			num_workers=self.num_workers,
			pin_memory=True,
			**dl_kwargs,
		)


__all__ = ["WireframeDataModule"]
