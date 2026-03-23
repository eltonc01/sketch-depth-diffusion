"""Training module export.

This thin wrapper keeps public import paths stable while the original
monolithic training file is incrementally split.
"""

from sketch_recon.training.train_diffusion import SketchDenoiserModule

__all__ = ["SketchDenoiserModule"]
