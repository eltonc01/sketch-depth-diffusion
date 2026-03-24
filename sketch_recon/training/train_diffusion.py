import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import torch.nn.functional as F
from sketch_recon.training.data.dataset import OcclusionAwareSequentialDataset
import multiprocessing as mp
import itertools
from datetime import timedelta
from sketch_recon.models.vae_unet_control import VAE, DenoisingUNet2DConditionModel, ControlEncoder
from tqdm import tqdm
import torch.distributed as dist
from sklearn.utils import shuffle

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from transformers import AutoModelForDepthEstimation, AutoImageProcessor, AutoConfig, AutoModel, get_linear_schedule_with_warmup
import time

from sketch_recon.training.control_encoders import DinoV2ControlEncoder, ViTSmallScratchControlEncoder
from sketch_recon.training.datamodule import WireframeDataModule
from sketch_recon.training.diffusion_utils import (
    absolute_loss,
    compute_latent_stats,
    gradient_loss,
    make_cosine_schedule,
    masked_tv_score,
)
from sketch_recon.config.checkpoints import resolve_model_variant_paths

NEAR = 3.0
FAR = 7.0

class SketchDenoiserModule(pl.LightningModule):
    def __init__(self, vae: VAE, unet: DenoisingUNet2DConditionModel, latent_stats_path: str,
                 img_size: int = 256, lr: float = 1e-4, weight_decay: float = 0.0, snr_weight: bool = True,
                 cfg_drop_p: float = 0.15, T: int = 1000, sketch_in_channels: int = 3,
                 use_controlnet: bool = False,
                 control_encoder_type: str = "vae",
                 dinov2_model_name: str = "facebook/dinov2-small",
                 train_dinov2_backbone: bool = True,
                 control_lr: float | None = None,
                 lambda_grad: float = 1.0,
                 struct_t_max: int = 200,
                 empty_cache_each_epoch: bool = True,
                 ):
        super().__init__()
        self.save_hyperparameters(ignore=['vae', 'unet'])
        self.vae = vae
        self.unet = unet
        self.img_size = img_size
        self.snr_weight = snr_weight
        self.cfg_drop_p = cfg_drop_p
        self.use_controlnet = use_controlnet
        self.control_encoder_type = control_encoder_type
        self.lambda_grad = float(lambda_grad)
        self.struct_t_max = int(struct_t_max)
        self.empty_cache_each_epoch = bool(empty_cache_each_epoch)
        # Use a smaller LR for the conditioning encoder by default
        self.control_lr = float(control_lr) if control_lr is not None else float(lr) * 0.1

        # Freeze VAE
        for p in self.vae.parameters():
            p.requires_grad = False
        self.vae.eval()

        # Create sketch encoder and setup conditioning
        if use_controlnet:
            if control_encoder_type == "dinov2":
                # DinoV2 ControlNet mode: learn a small 2->3ch adapter and optionally fine-tune DinoV2.
                # We extract features directly at decoder resolutions (32, 16, 8) to minimize interpolation.
                self.control_encoder = DinoV2ControlEncoder(
                    model_name=dinov2_model_name,
                    out_channels=128,
                    target_resolutions=(32, 16, 8),
                    train_backbone=train_dinov2_backbone,
                    in_channels=sketch_in_channels,
                )
            elif control_encoder_type == "vit_small_scratch":
                self.control_encoder = ViTSmallScratchControlEncoder(
                    model_name="vit_small_patch16_224",
                    out_channels=128,
                    target_resolutions=(32, 16, 8),
                    in_channels=sketch_in_channels,
                )
            elif control_encoder_type == "vae":
                # VAE ControlNet mode: Create encoder and immediately wrap in ControlEncoder
                # Don't store sketch_encoder separately to avoid unused parameters in DDP
                sketch_encoder = self.vae.duplicate_encoder_for_sketch(sketch_in_channels=sketch_in_channels)
                
                # Set requires_grad in ONE clean pass to avoid DDP race conditions
                # Make all parameters trainable EXCEPT the unused final layers
                for name, p in sketch_encoder.named_parameters():
                    if 'norm_out' in name or 'mu_logvar' in name:
                        # Freeze unused final layers (ControlNet doesn't use them)
                        p.requires_grad = False
                    else:
                        # Train the rest (init, blocks, etc.)
                        p.requires_grad = True
                
                # For 32x32 latent space: extract at 128, 64, 32 (VAE encoder: 256->128->64->32)
                self.control_encoder = ControlEncoder(sketch_encoder, extract_resolutions=(128, 64, 32))
                # ControlEncoder now owns the sketch_encoder, all gradients flow through it
            else:
                raise ValueError(
                    f"Unknown control_encoder_type: {control_encoder_type}. "
                    "Use 'vae', 'dinov2', or 'vit_small_scratch'."
                )
        else:
            # Standard mode: Store sketch encoder for concatenation-based conditioning
            self.sketch_encoder = self.vae.duplicate_encoder_for_sketch(sketch_in_channels=sketch_in_channels)
            for p in self.sketch_encoder.parameters():
                p.requires_grad = True
            self.control_encoder = None

        # Train only UNet and conditioning modules (sketch_encoder or control_encoder)
        for p in self.unet.parameters():
            p.requires_grad = True

        # Diffusion schedule
        betas, alphas, abar = make_cosine_schedule(T=T)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('abar', abar)

        # Latent normalization stats
        latent_stats = torch.load(latent_stats_path, map_location=torch.device("cpu"))
        self.register_buffer('latent_mean_vec', latent_stats['mean'])
        self.register_buffer('latent_std_vec', latent_stats['std'])

    @torch.no_grad()
    def encode_latent(self, gt_depth: torch.Tensor) -> torch.Tensor:
        # gt_depth: (B,1,H,W) -> resize to 256 and encode with VAE
        depth_resized = F.interpolate(gt_depth, size=(self.img_size, self.img_size), mode='bilinear', align_corners=True)
        mu, _ = self.vae.encoder(depth_resized)
        mu_n = (mu - self.latent_mean_vec.view(1, -1, 1, 1)) / self.latent_std_vec.view(1, -1, 1, 1)
        return mu_n

    def _prepare_sketch(self, imgs: torch.Tensor) -> torch.Tensor:
        # imgs: (B,1,H,W) -> resize to 256 for sketch encoder (same as VAE)
        return F.interpolate(imgs, size=(self.img_size, self.img_size), mode='bilinear', align_corners=True)

    @torch.no_grad()
    def encode_sketch(self, sketch: torch.Tensor) -> torch.Tensor:
        """Encode sketch to latent space (only available in standard mode, not ControlNet)."""
        if self.use_controlnet:
            raise RuntimeError("encode_sketch() not available in ControlNet mode. Use control_encoder for feature extraction.")
        sketch_resized = F.interpolate(sketch, size=(self.img_size, self.img_size), mode='bilinear', align_corners=True)
        mu, _ = self.sketch_encoder(sketch_resized)
        mu_n = (mu - self.latent_mean_vec.view(1, -1, 1, 1)) / self.latent_std_vec.view(1, -1, 1, 1)
        return mu_n

    def training_step(self, batch, batch_idx):
        imgs, gt_depth, noise_scalars = batch  # imgs: (B,2,H,W) = (sketch, partial_hint)
        imgs = imgs.to(self.device)
        gt_depth = gt_depth.to(self.device)
        
        # Handle noise_scalars: if already tensor, move to device; if list/tuple, convert
        if isinstance(noise_scalars, (list, tuple)):
            noise_scalars = torch.tensor(noise_scalars, dtype=torch.float32, device=self.device)
        else:
            noise_scalars = noise_scalars.clone().detach().to(self.device).float()

        with torch.no_grad():
            z0n = self.encode_latent(gt_depth)  # (B,4,16,16)

        # Encode sketch with trainable sketch encoder / control encoder
        sketch_resized = F.interpolate(imgs, size=(self.img_size, self.img_size), mode='bilinear', align_corners=True)
        
        # Get control features if using ControlNet, otherwise just get sketch latent
        if self.use_controlnet:
            control_features = self.control_encoder(sketch_resized)  # {128: feat128, 64: feat64, 32: feat32}
            # ControlNet mode: No concatenation, so sketch_latent not needed for UNet input
            sketch_latent = None
        else:
            control_features = None
            sketch_mu, _ = self.sketch_encoder(sketch_resized)
            sketch_latent = (sketch_mu - self.latent_mean_vec.view(1, -1, 1, 1)) / self.latent_std_vec.view(1, -1, 1, 1)

        # Classifier-free guidance dropout: drop conditioning with probability cfg_drop_p
        B = imgs.size(0)
        cfg_mask = None
        if self.cfg_drop_p > 0:
            # IMPORTANT: In DDP, we need deterministic CFG dropout across all GPUs
            # Using batch_idx as seed ensures all GPUs drop the same samples
            # This prevents gradient synchronization mismatches that cause hanging
            generator = torch.Generator(device=self.device)
            generator.manual_seed(batch_idx + self.current_epoch * 100000)
            cfg_mask = (torch.rand(B, device=self.device, generator=generator) >= self.cfg_drop_p).float().view(B, 1, 1, 1)
            
            if self.use_controlnet:
                # ControlNet: Only mask control features (no sketch_latent to mask)
                control_features_dropped = {}
                for k, v in control_features.items():
                    # Clone and mask to preserve autograd graph
                    control_features_dropped[k] = v.clone() * cfg_mask
                sketch_latent_dropped = None
            else:
                # Standard mode: Mask sketch_latent for concatenation
                sketch_latent_dropped = sketch_latent * cfg_mask
                control_features_dropped = None
        else:
            sketch_latent_dropped = sketch_latent
            control_features_dropped = control_features

        t = torch.randint(0, self.abar.numel(), (B,), device=self.device)
        abar_t = self.abar[t].view(B, 1, 1, 1)
        eps = torch.randn_like(z0n)
        zt = abar_t.sqrt() * z0n + (1.0 - abar_t).sqrt() * eps

        # v-parameterization target and loss
        v_true = abar_t.sqrt() * eps - (1.0 - abar_t).sqrt() * z0n
        v_hat = self.unet(zt, t, sketch_latent_dropped, control_features=control_features_dropped, noise_scalar=noise_scalars)
        v_mse = F.mse_loss(v_hat, v_true, reduction='none').mean(dim=[1, 2, 3])

        if self.snr_weight:
            snr = abar_t / (1 - abar_t + 1e-8)
            weights = snr / (snr + 1)
            loss = (weights * v_mse).mean()
        else:
            loss = v_mse.mean()

        # Structural auxiliary loss on decoded x0 prediction (masked to stroke pixels).
        # Keep simple: only apply at late timesteps and only if enabled.
        grad_struct_loss = torch.tensor(0.0, device=self.device)
        if self.lambda_grad > 0.0 and self.struct_t_max > 0:
            late_idx = (t <= self.struct_t_max).nonzero(as_tuple=False).squeeze(-1)
            if late_idx.numel() > 0:
                abar_t_sub = abar_t[late_idx]
                zt_sub = zt[late_idx]
                v_hat_sub = v_hat[late_idx]
                x0_hat_n_sub = abar_t_sub.sqrt() * zt_sub - (1.0 - abar_t_sub).sqrt() * v_hat_sub
                z_denorm_hat_sub = x0_hat_n_sub * self.latent_std_vec.view(1, -1, 1, 1) + self.latent_mean_vec.view(1, -1, 1, 1)
                depth_hat_sub = self.vae.decoder(z_denorm_hat_sub)
                if depth_hat_sub.shape[-2:] != gt_depth.shape[-2:]:
                    depth_hat_sub = F.interpolate(depth_hat_sub, size=gt_depth.shape[-2:], mode='bilinear', align_corners=True)

                stroke_mask_sub = (imgs[late_idx, 0:1] > 0.5)
                grad_struct_loss = gradient_loss(depth_hat_sub, gt_depth[late_idx], mask=stroke_mask_sub).mean()
                loss = loss + self.lambda_grad * grad_struct_loss

        # Logging: avoid per-step DDP sync (expensive). Sync only on epoch.
        self.log('train_diff_step', loss, on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
        self.log('train_diff', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_v_mse', v_mse.mean(), on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
        if self.lambda_grad > 0.0 and self.struct_t_max > 0:
            self.log('train_grad_struct_step', grad_struct_loss, on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
            self.log('train_grad_struct', grad_struct_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        
        # Log separate metrics for conditional vs unconditional samples
        if cfg_mask is not None:
            cond_mask = cfg_mask.squeeze()  # (B,)
            uncond_mask = 1.0 - cond_mask
            
            if cond_mask.sum() > 0:
                if self.snr_weight:
                    cond_loss = (weights.squeeze() * v_mse)[cond_mask > 0.5].mean()
                else:
                    cond_loss = v_mse[cond_mask > 0.5].mean()
                self.log('train_diff_conditional_step', cond_loss, on_step=True, on_epoch=False, sync_dist=False)
                self.log('train_diff_conditional', cond_loss, on_step=False, on_epoch=True, sync_dist=True)
            
            if uncond_mask.sum() > 0:
                if self.snr_weight:
                    uncond_loss = (weights.squeeze() * v_mse)[uncond_mask > 0.5].mean()
                else:
                    uncond_loss = v_mse[uncond_mask > 0.5].mean()
                self.log('train_diff_unconditional_step', uncond_loss, on_step=True, on_epoch=False, sync_dist=False)
                self.log('train_diff_unconditional', uncond_loss, on_step=False, on_epoch=True, sync_dist=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, gt_depth, noise_scalars = batch
        imgs = imgs.to(self.device)
        gt_depth = gt_depth.to(self.device)
        
        # Handle noise_scalars: if already tensor, move to device; if list/tuple, convert
        if isinstance(noise_scalars, (list, tuple)):
            noise_scalars = torch.tensor(noise_scalars, dtype=torch.float32, device=self.device)
        else:
            noise_scalars = noise_scalars.clone().detach().to(self.device).float()

        with torch.no_grad():
            z0n = self.encode_latent(gt_depth)
            
            # Encode sketch conditioning input
            sketch_resized = F.interpolate(imgs, size=(self.img_size, self.img_size), mode='bilinear', align_corners=True)
            
            # Get control features if using ControlNet
            if self.use_controlnet:
                control_features = self.control_encoder(sketch_resized)
                sketch_latent = None  # ControlNet mode: no concatenation
            else:
                control_features = None
                sketch_mu, _ = self.sketch_encoder(sketch_resized)
                sketch_latent = (sketch_mu - self.latent_mean_vec.view(1, -1, 1, 1)) / self.latent_std_vec.view(1, -1, 1, 1)

            B = imgs.size(0)
            t = torch.randint(0, self.abar.numel(), (B,), device=self.device)
            abar_t = self.abar[t].view(B, 1, 1, 1)
            eps = torch.randn_like(z0n)
            zt = abar_t.sqrt() * z0n + (1.0 - abar_t).sqrt() * eps

            # v-parameterization target and loss
            v_true = abar_t.sqrt() * eps - (1.0 - abar_t).sqrt() * z0n
            v_hat = self.unet(zt, t, sketch_latent, control_features=control_features, noise_scalar=noise_scalars)
            v_mse = F.mse_loss(v_hat, v_true, reduction='none').mean(dim=[1, 2, 3])

            if self.snr_weight:
                snr = abar_t / (1 - abar_t + 1e-8)
                weights = snr / (snr + 1)
                loss = (weights * v_mse).mean()
            else:
                loss = v_mse.mean()

            base_loss = loss

            grad_struct_loss = torch.tensor(0.0, device=self.device)
            if self.lambda_grad > 0.0 and self.struct_t_max > 0:
                late_idx = (t <= self.struct_t_max).nonzero(as_tuple=False).squeeze(-1)
                if late_idx.numel() > 0:
                    abar_t_sub = abar_t[late_idx]
                    zt_sub = zt[late_idx]
                    v_hat_sub = v_hat[late_idx]
                    x0_hat_n_sub = abar_t_sub.sqrt() * zt_sub - (1.0 - abar_t_sub).sqrt() * v_hat_sub
                    z_denorm_hat_sub = x0_hat_n_sub * self.latent_std_vec.view(1, -1, 1, 1) + self.latent_mean_vec.view(1, -1, 1, 1)
                    depth_hat_sub = self.vae.decoder(z_denorm_hat_sub)
                    if depth_hat_sub.shape[-2:] != gt_depth.shape[-2:]:
                        depth_hat_sub = F.interpolate(depth_hat_sub, size=gt_depth.shape[-2:], mode='bilinear', align_corners=True)

                    stroke_mask_sub = (imgs[late_idx, 0:1] > 0.5)
                    grad_struct_loss = gradient_loss(depth_hat_sub, gt_depth[late_idx], mask=stroke_mask_sub).mean()
                    loss = loss + self.lambda_grad * grad_struct_loss

        self.log('val_diff', loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_diff_base', base_loss, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('val_v_mse', v_mse.mean(), on_epoch=True, prog_bar=False, sync_dist=True)
        if self.lambda_grad > 0.0 and self.struct_t_max > 0:
            self.log('val_grad_struct', grad_struct_loss, on_epoch=True, prog_bar=False, sync_dist=True)

    @torch.no_grad()
    def inference(self, sketch: torch.Tensor, num_steps: int = 50, cfg_scale: float = 3.0, 
                  use_ddim: bool = True, eta: float = 0.0, return_intermediates: bool = False,
                  noise_scalar: float = None, clamp_known_pixels: bool = True,
                  inpaint_known_pixels: bool = True,
                  num_candidates: int = 1,
                  select_by_structure: bool = True) -> torch.Tensor:
        """Generate depth predictions from sketch input using DDPM/DDIM sampling.
        
        Args:
            sketch: (B, C, H, W) sketch input tensor - will be resized to 256x256 internally
            num_steps: Number of denoising steps (default: 50)
            cfg_scale: Classifier-free guidance scale (default: 3.0). Set to 1.0 to disable.
            use_ddim: If True, use DDIM sampling; otherwise use DDPM (default: True)
            eta: DDIM stochasticity parameter, 0=deterministic, 1=DDPM (default: 0.0)
            return_intermediates: If True, return list of intermediate predictions (default: False)
            noise_scalar: Optional normalized noise level (0.0-1.0). If None, assumes max noise (1.0)
                         0.0 = clean sketch (no fixing needed)
                         1.0 = max noise (aggressive fixing, 25% jitter strength)
            clamp_known_pixels: If True, enforce trusted hint pixels in the final decoded
                               prediction using the valid-mask channel (Option B).
            inpaint_known_pixels: If True, enforce known pixels during every denoising
                                 step (inpainting-style latent constraint).
            
        Returns:
            depth: (B, 1, H, W) predicted depth map in [0, 1] range, same resolution as input sketch
            OR
            (depth, intermediates): tuple if return_intermediates=True
        """
        if int(num_candidates) > 1:
            if return_intermediates:
                raise ValueError("return_intermediates=True is not supported with num_candidates > 1")

            candidates = []
            for _ in range(int(num_candidates)):
                cand = self.inference(
                    sketch=sketch,
                    num_steps=num_steps,
                    cfg_scale=cfg_scale,
                    use_ddim=use_ddim,
                    eta=eta,
                    return_intermediates=False,
                    noise_scalar=noise_scalar,
                    clamp_known_pixels=clamp_known_pixels,
                    inpaint_known_pixels=inpaint_known_pixels,
                    num_candidates=1,
                    select_by_structure=select_by_structure,
                )
                candidates.append(cand)

            candidates_stacked = torch.stack(candidates, dim=0)  # (K, B, 1, H, W)
            if not select_by_structure:
                return candidates_stacked.mean(dim=0)

            # Score each candidate by masked TV inside stroke pixels (lower is better).
            stroke_mask = (sketch[:, 0:1] > 0.5).float()
            scores = torch.stack([masked_tv_score(cand, stroke_mask) for cand in candidates], dim=0)  # (K, B)
            best_idx = torch.argmin(scores, dim=0)  # (B,)
            best = torch.stack([candidates_stacked[int(best_idx[b]), b] for b in range(candidates_stacked.shape[1])], dim=0)
            return best

        self.eval()
        device = sketch.device
        B, _, H_orig, W_orig = sketch.shape
        
        # Default noise scalar to maximum (1.0) if not provided (assume worst-case noise)
        if noise_scalar is None:
            noise_scalar = 1.0  # Max noise in normalized [0, 1] range
        noise_scalar_tensor = torch.full((B,), noise_scalar, dtype=torch.float32, device=device)
        
        # Prepare sketch for encoder: ensure it has the expected number of channels.
        expected_c = int(getattr(self.hparams, "sketch_in_channels", 3))
        if sketch.shape[1] < expected_c:
            pad_c = expected_c - sketch.shape[1]
            sketch = torch.cat([sketch, torch.zeros((B, pad_c, H_orig, W_orig), device=device, dtype=sketch.dtype)], dim=1)
        sketch_c = sketch[:, :expected_c, :, :]

        # Option B channels: [sketch, partial_depth_hint, hint_valid].
        # valid==1 means the hint value is fully trustworthy and should be copied through.
        known_depth = None
        known_valid = None
        if sketch_c.shape[1] >= 3:
            known_depth = sketch_c[:, 1:2]
            known_valid = sketch_c[:, 2:3] > 0.5

        sketch_resized = F.interpolate(sketch_c, size=(self.img_size, self.img_size), mode='bilinear', align_corners=True)
        
        # Get control features if using ControlNet
        if self.use_controlnet:
            sketch_latent_cond_features = self.control_encoder(sketch_resized)
            sketch_latent_cond = None  # ControlNet mode: no concatenation
        else:
            sketch_latent_cond_features = None
            sketch_mu, _ = self.sketch_encoder(sketch_resized)
            sketch_latent_cond = (sketch_mu - self.latent_mean_vec.view(1, -1, 1, 1)) / self.latent_std_vec.view(1, -1, 1, 1)
        
        # Get latent shape from VAE (should be 32x32 for init_res=32)
        with torch.no_grad():
            # Use a dummy depth to get latent shape
            dummy = torch.zeros(1, 1, self.img_size, self.img_size, device=device)
            mu, _ = self.vae.encoder(dummy)
            _, C, H_lat, W_lat = mu.shape
        
        # Initialize with random noise
        zt = torch.randn(B, C, H_lat, W_lat, device=device)

        # Option B inpainting-style setup: constrain trusted pixels throughout denoising.
        inpaint_active = bool(inpaint_known_pixels and (known_depth is not None) and (known_valid is not None))
        known_valid_lat = None
        z0_known_n = None
        eps_known = None
        if inpaint_active:
            known_depth_rs = known_depth
            known_valid_rs = known_valid
            if known_depth_rs.shape[-2:] != (self.img_size, self.img_size):
                known_depth_rs = F.interpolate(known_depth_rs, size=(self.img_size, self.img_size), mode='bilinear', align_corners=True)
                known_valid_rs = F.interpolate(known_valid_rs.float(), size=(self.img_size, self.img_size), mode='nearest') > 0.5

            known_depth_rs = torch.where(
                known_valid_rs,
                torch.clamp(known_depth_rs, 0.0, 1.0),
                torch.zeros_like(known_depth_rs),
            )

            known_mu, _ = self.vae.encoder(known_depth_rs)
            z0_known_n = (known_mu - self.latent_mean_vec.view(1, -1, 1, 1)) / self.latent_std_vec.view(1, -1, 1, 1)
            known_valid_lat = F.interpolate(known_valid_rs.float(), size=(H_lat, W_lat), mode='nearest') > 0.5
            eps_known = torch.randn_like(zt)

            def _apply_inpainting_constraint(z_state: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
                abar_state = self.abar[t_idx].view(1, 1, 1, 1)
                z_known_t = abar_state.sqrt() * z0_known_n + (1.0 - abar_state).sqrt() * eps_known
                return torch.where(known_valid_lat, z_known_t, z_state)
        
        # Set up timestep schedule
        T_total = len(self.abar)
        if use_ddim:
            # DDIM: use subset of timesteps
            timesteps = torch.linspace(T_total - 1, 0, num_steps, dtype=torch.long, device=device)
        else:
            # DDPM: use all timesteps or evenly spaced subset
            if num_steps >= T_total:
                timesteps = torch.arange(T_total - 1, -1, -1, dtype=torch.long, device=device)
            else:
                timesteps = torch.linspace(T_total - 1, 0, num_steps, dtype=torch.long, device=device)
        
        intermediates = [] if return_intermediates else None
        
        # Denoising loop
        for i, t_curr in enumerate(timesteps):
            t_batch = t_curr.repeat(B)
            
            # Classifier-free guidance: run model twice (conditional and unconditional)
            if cfg_scale != 1.0:
                # Conditional prediction
                v_cond = self.unet(zt, t_batch, sketch_latent_cond, control_features=sketch_latent_cond_features, 
                                   noise_scalar=noise_scalar_tensor)
                
                # Unconditional prediction
                if self.use_controlnet:
                    # ControlNet mode: zero out control features only
                    control_features_uncond = {k: torch.zeros_like(v) for k, v in sketch_latent_cond_features.items()}
                    v_uncond = self.unet(zt, t_batch, None, control_features=control_features_uncond, 
                                        noise_scalar=noise_scalar_tensor)
                else:
                    # Standard mode: zero out sketch latent
                    sketch_latent_uncond = torch.zeros_like(sketch_latent_cond)
                    v_uncond = self.unet(zt, t_batch, sketch_latent_uncond, control_features=None, 
                                        noise_scalar=noise_scalar_tensor)
                
                # Apply guidance
                v_hat = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                # No guidance
                v_hat = self.unet(zt, t_batch, sketch_latent_cond, control_features=sketch_latent_cond_features,
                                 noise_scalar=noise_scalar_tensor)
            
            # Get alpha values
            abar_t = self.abar[t_curr].view(1, 1, 1, 1)
            
            if use_ddim:
                # DDIM update
                # Predict x0 from v-parameterization: x0 = sqrt(abar_t) * zt - sqrt(1-abar_t) * v
                x0_pred = abar_t.sqrt() * zt - (1.0 - abar_t).sqrt() * v_hat
                
                # Predict eps from v-parameterization: v = sqrt(abar_t) * eps - sqrt(1-abar_t) * x0
                # Therefore: eps = (v + sqrt(1-abar_t) * x0) / sqrt(abar_t)
                eps_pred = (v_hat + (1.0 - abar_t).sqrt() * x0_pred) / (abar_t.sqrt() + 1e-8)
                
                if i < len(timesteps) - 1:
                    # Not the last step
                    t_next = timesteps[i + 1]
                    abar_next = self.abar[t_next].view(1, 1, 1, 1)
                    
                    # DDIM formula with stochasticity parameter eta
                    sigma_t = eta * torch.sqrt((1 - abar_next) / (1 - abar_t + 1e-8) * (1 - abar_t / (abar_next + 1e-8)))
                    
                    # Direction pointing to zt (using predicted noise, not v)
                    dir_zt = torch.sqrt(1 - abar_next - sigma_t**2 + 1e-8) * eps_pred
                    
                    # Random noise
                    noise = torch.randn_like(zt) if eta > 0 else 0
                    
                    # Update
                    zt = abar_next.sqrt() * x0_pred + dir_zt + sigma_t * noise
                else:
                    # Last step: return x0 prediction
                    zt = x0_pred
                    
            else:
                # DDPM update
                # Predict x0 from v-parameterization                x0_pred = abar_t.sqrt() * zt - (1.0 - abar_t).sqrt() * v_hat
                
                if t_curr > 0:
                    # Not the last step
                    alpha_t = self.alphas[t_curr]
                    abar_prev = self.abar[t_curr - 1] if t_curr > 0 else torch.tensor(1.0, device=device)
                    
                    # Posterior mean
                    coef1 = (abar_prev.sqrt() * self.betas[t_curr]) / (1 - abar_t + 1e-8)
                    coef2 = ((1 - abar_prev) * alpha_t.sqrt()) / (1 - abar_t + 1e-8)
                    mu = coef1 * x0_pred + coef2 * zt
                    
                    # Posterior variance
                    sigma = torch.sqrt(self.betas[t_curr] * (1 - abar_prev) / (1 - abar_t + 1e-8))
                    
                    # Sample
                    noise = torch.randn_like(zt)
                    zt = mu + sigma * noise
                else:
                    # Last step
                    zt = x0_pred

            if inpaint_active:
                if i < len(timesteps) - 1:
                    zt = _apply_inpainting_constraint(zt, timesteps[i + 1])
                else:
                    # Final state corresponds to x0; enforce exact known latent on trusted pixels.
                    zt = torch.where(known_valid_lat, z0_known_n, zt)
            
            # Store intermediate if requested
            if return_intermediates and i % max(1, len(timesteps) // 10) == 0:
                with torch.no_grad():
                    z_denorm = zt * self.latent_std_vec.view(1, -1, 1, 1) + self.latent_mean_vec.view(1, -1, 1, 1)
                    depth_pred = self.vae.decoder(z_denorm)
                    depth_pred = F.interpolate(depth_pred, size=(H_orig, W_orig), mode='bilinear', align_corners=True)
                    if clamp_known_pixels and (known_depth is not None) and (known_valid is not None):
                        known_depth_rs = known_depth
                        known_valid_rs = known_valid
                        if known_depth_rs.shape[-2:] != depth_pred.shape[-2:]:
                            known_depth_rs = F.interpolate(known_depth_rs, size=depth_pred.shape[-2:], mode='bilinear', align_corners=True)
                            known_valid_rs = F.interpolate(known_valid_rs.float(), size=depth_pred.shape[-2:], mode='nearest') > 0.5
                        depth_pred = torch.where(known_valid_rs, torch.clamp(known_depth_rs, 0.0, 1.0), depth_pred)
                    depth_pred = torch.clamp(depth_pred, 0, 1)
                    intermediates.append(depth_pred.cpu())
        
        # Denormalize latent and decode to depth
        z_denorm = zt * self.latent_std_vec.view(1, -1, 1, 1) + self.latent_mean_vec.view(1, -1, 1, 1)
        depth_pred = self.vae.decoder(z_denorm)
        
        # Resize to original input resolution
        depth_pred = F.interpolate(depth_pred, size=(H_orig, W_orig), mode='bilinear', align_corners=True)
        
        # Clamp to valid range
        depth_pred = torch.clamp(depth_pred, 0, 1)

        if clamp_known_pixels and (known_depth is not None) and (known_valid is not None):
            known_depth_rs = known_depth
            known_valid_rs = known_valid
            if known_depth_rs.shape[-2:] != depth_pred.shape[-2:]:
                known_depth_rs = F.interpolate(known_depth_rs, size=depth_pred.shape[-2:], mode='bilinear', align_corners=True)
                known_valid_rs = F.interpolate(known_valid_rs.float(), size=depth_pred.shape[-2:], mode='nearest') > 0.5
            depth_pred = torch.where(known_valid_rs, torch.clamp(known_depth_rs, 0.0, 1.0), depth_pred)
        
        if return_intermediates:
            return depth_pred, intermediates
        return depth_pred

    def configure_optimizers(self):
        # Separate parameter groups with different learning rates
        param_groups = []
        
        # UNet parameters
        unet_params = [p for p in self.unet.parameters() if p.requires_grad]
        param_groups.append({'params': unet_params, 'lr': self.hparams.lr})
        
        # Conditioning encoder gets a smaller learning rate (often helps stability)
        if self.use_controlnet:
            # ControlNet: train control encoder (which wraps sketch encoder)
            control_encoder_params = [p for p in self.control_encoder.parameters() if p.requires_grad]
            param_groups.append({'params': control_encoder_params, 'lr': self.control_lr})
        else:
            # Original: train sketch encoder directly
            sketch_encoder_params = [p for p in self.sketch_encoder.parameters() if p.requires_grad]
            param_groups.append({'params': sketch_encoder_params, 'lr': self.control_lr})
        
        optimizer = optim.AdamW(param_groups, weight_decay=self.hparams.weight_decay)
        
        return optimizer

    def on_train_epoch_end(self):
        if self.empty_cache_each_epoch and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_validation_epoch_end(self):
        if self.empty_cache_each_epoch and torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    def _parse_int_tuple(arg: str, name: str) -> tuple[int, ...]:
        try:
            values = tuple(int(x.strip()) for x in str(arg).split(",") if x.strip() != "")
        except Exception as e:
            raise ValueError(f"Invalid {name}: {arg}") from e
        if not values:
            raise ValueError(f"{name} must contain at least one integer")
        return values

    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", type=int, nargs='+', required=False, default=0, help="GPU device indices")
    parser.add_argument("--epochs", type=int, default=100, help="Total number of epochs to train.")
    parser.add_argument("--checkpoint_path", type=str, default="depth_anything_checkpoints", help="Path to save checkpoints.")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Path to resume from a checkpoint.")
    parser.add_argument("--log_version", type=str, default=None, help="Version to resume logging (e.g., 'version_0'). If None, creates new version.")
    parser.add_argument("--checkpoint_manifest", type=str, default=None, help="Optional path to checkpoint manifest JSON.")
    parser.add_argument("--artifacts_manifest", dest="checkpoint_manifest", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--model_variant", type=str, default="dinov2_vast", help="Model variant key in checkpoint manifest.")
    parser.add_argument("--vae_checkpoint", type=str, default=None, help="Path to pretrained VAE checkpoint.")
    parser.add_argument("--latent_stats_path", type=str, default=None, help="Path to latent stats (.pth).")
    parser.add_argument("--threshold", type=int, default=None, help="Filter number of edges")
    
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1, 
                        help="Number of batches to accumulate gradients over. "
                             "Effective batch size = batch_size * accumulate_grad_batches * num_gpus")
    parser.add_argument("--decoder_lr", type=float, default=1e-4, help="Learning rate for the decoder.")
    parser.add_argument("--encoder_lr", type=float, default=1e-5, help="Learning rate for the conditioning encoder (ControlNet encoder / sketch encoder).")
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--decoder_epochs", type=int, default=0, help="Number of epochs to train only the decoder.")

    parser.add_argument("--reg_dir", type=str, default="abc/reg")
    parser.add_argument("--depth_dir", type=str, default="abc/depth")
    parser.add_argument("--workers", type=int, default=4)

    # Runtime / perf knobs (helps portability across GPU types)
    _bool_action = getattr(argparse, "BooleanOptionalAction", None)
    if _bool_action is None:
        raise RuntimeError(
            "This script requires argparse.BooleanOptionalAction (Python 3.9+). "
            "Please upgrade Python or update the CLI flags for booleans."
        )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16-mixed",
        choices=["16-mixed", "bf16-mixed", "32-true"],
        help=(
            "Lightning precision mode. On RTX 4090, bf16-mixed is often faster/more stable than 16-mixed. "
            "(2080-class GPUs do not support bf16.)"
        ),
    )
    parser.add_argument(
        "--torch_compile",
        action=_bool_action,
        default=False,
        help="Use torch.compile on the UNet (can improve throughput on Ampere/Ada; may increase startup time).",
    )
    parser.add_argument(
        "--torch_compile_mode",
        type=str,
        default="max-autotune",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode (only used with --torch_compile).",
    )
    parser.add_argument(
        "--channels_last",
        action=_bool_action,
        default=True,
        help="Use channels_last memory format for conv-heavy models (often faster on 4090s).",
    )
    parser.add_argument(
        "--log_every_n_steps",
        type=int,
        default=50,
        help="Logging cadence (reduce to lower overhead on multi-GPU).",
    )

    # UNet capacity knobs (defaults tuned for ~120M-180M UNet range)
    parser.add_argument(
        "--unet_model_channels",
        type=int,
        default=192,
        help="Base channel width for diffusion UNet (128 baseline; 192 is larger).",
    )
    parser.add_argument(
        "--unet_num_res_blocks",
        type=int,
        default=2,
        help="Residual blocks per UNet stage.",
    )
    parser.add_argument(
        "--unet_channel_mult",
        type=str,
        default="1,2,4",
        help="Comma-separated UNet channel multipliers per stage.",
    )
    parser.add_argument(
        "--unet_attn_resolutions",
        type=str,
        default="8,16,32",
        help="Comma-separated decoder resolutions that use attention.",
    )
    
    # ControlNet configuration (default ON)
    parser.add_argument(
        "--use_controlnet",
        action=_bool_action,
        default=True,
        help="Enable ControlNet-style spatial feature injection (default: enabled).",
    )
    parser.add_argument(
        "--control_encoder",
        type=str,
        choices=["vae", "dinov2", "vit_small_scratch"],
        default="vae",
        help="ControlNet encoder backbone for sketch conditioning.",
    )
    parser.add_argument(
        "--dinov2_model",
        type=str,
        default="facebook/dinov2-base",
        help="HuggingFace model id for DinoV2 (used when --control_encoder=dinov2).",
    )
    parser.add_argument(
        "--train_dinov2_backbone",
        action=_bool_action,
        default=True,
        help="Fine-tune DinoV2 backbone (default: enabled). Use --no-train_dinov2_backbone to freeze.",
    )
    
    # Data mode configuration
    parser.add_argument(
        "--data_mode",
        type=str,
        choices=["clean", "noisy"],
        default="clean",
        help=(
            "Explicit dataset mode gate. clean=perfect_to_perfect, "
            "noisy=imperfect_to_perfect."
        ),
    )
    parser.add_argument(
        "--use_noisy_data",
        action="store_true",
        help=(
            "Deprecated compatibility flag. Equivalent to --data_mode noisy. "
            "Will be removed in a future cleanup pass."
        ),
    )

    parser.add_argument(
        "--occlusion_aware_partial_depth",
        action=_bool_action,
        default=True,
        help=(
            "If enabled (default), partial-depth hints may include occluded/behind-surface values. "
            "Model conditioning uses Option B: 3-channel [sketch, partial_depth, hint_valid], "
            "where hint_valid==1 marks 100%% trustworthy hint pixels."
        ),
    )
    parser.add_argument(
        "--lambda_grad",
        type=float,
        default=1.0,
        help="Weight for masked gradient-matching structural loss on decoded x0 (0 disables).",
    )
    parser.add_argument(
        "--struct_t_max",
        type=int,
        default=200,
        help="Apply structural loss only for timesteps t <= struct_t_max.",
    )
    parser.add_argument(
        "--empty_cache_each_epoch",
        action=_bool_action,
        default=True,
        help="Call torch.cuda.empty_cache() at train/val epoch boundaries to reduce fragmentation.",
    )
    args = parser.parse_args()

    # Explicit mode gate with backward compatibility.
    if args.use_noisy_data and args.data_mode == "clean":
        args.data_mode = "noisy"

    use_noisy_data = (args.data_mode == "noisy")

    if args.vae_checkpoint is None or args.latent_stats_path is None:
        resolved = resolve_model_variant_paths(
            model_variant=args.model_variant,
            manifest_path=args.checkpoint_manifest,
        )
        if args.vae_checkpoint is None:
            args.vae_checkpoint = resolved["vae_checkpoint"]
        if args.latent_stats_path is None:
            args.latent_stats_path = resolved["latent_stats_path"]

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    global_rank = int(os.environ.get("RANK", "0"))
    is_rank0 = (local_rank == 0 and global_rank == 0)

    if is_rank0:
        print("\n" + "="*70)
        print("DATA MODE GATE")
        print("="*70)
        print(f"  data_mode: {args.data_mode}")
        print(f"  use_noisy_data (resolved): {use_noisy_data}")
        print("="*70 + "\n")

    # Resolve runtime settings for current GPU architecture.
    # Ampere+ (SM80+) typically benefits from channels_last/bf16 and can use
    # gradient_as_bucket_view safely with our layout choices. Pre-Ampere often cannot.
    resolved_precision = args.precision
    resolved_channels_last = bool(args.channels_last)
    resolved_gradient_as_bucket_view = True
    resolved_torch_compile = bool(args.torch_compile)
    device_cc = None

    if torch.cuda.is_available():
        try:
            cc = torch.cuda.get_device_capability(torch.cuda.current_device())
            device_cc = (int(cc[0]), int(cc[1]))
        except Exception:
            device_cc = None

    is_ampere_or_newer = bool(device_cc is not None and device_cc[0] >= 8)

    # Pre-Ampere (e.g., RTX 2080 Ti, SM75): avoid settings that can cause
    # DDP grad-stride bucket warnings and reduced throughput.
    if torch.cuda.is_available() and (not is_ampere_or_newer):
        if resolved_precision == "bf16-mixed":
            resolved_precision = "16-mixed"
        resolved_channels_last = False
        resolved_gradient_as_bucket_view = False

        # torch.compile on older cards can regress throughput; keep user intent,
        # but disable by default unless they explicitly passed --torch_compile.
        # (args.torch_compile already reflects explicit CLI choice.)

    if is_rank0:
        print("\n" + "="*70)
        print("RUNTIME PERFORMANCE CONFIG")
        print("="*70)
        if device_cc is not None:
            print(f"  CUDA compute capability: {device_cc[0]}.{device_cc[1]}")
            print(f"  GPU class: {'Ampere+' if is_ampere_or_newer else 'Pre-Ampere'}")
        else:
            print("  CUDA compute capability: unavailable")
        print(f"  Precision: {resolved_precision}")
        print(f"  Channels-last: {resolved_channels_last}")
        print(f"  DDP gradient_as_bucket_view: {resolved_gradient_as_bucket_view}")
        print(f"  torch.compile: {resolved_torch_compile}")
        print("="*70 + "\n")

    # Performance knobs (safe defaults for fixed-size conv workloads)
    if torch.cuda.is_available():
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Structural loss uses variable-size sub-batches (late_idx), which can cause
        # cudnn benchmark workspace churn / fragmentation over time.
        # Disable benchmark in this case to improve memory stability.
        if args.lambda_grad > 0.0 and args.struct_t_max > 0:
            torch.backends.cudnn.benchmark = False
            if is_rank0:
                print("[perf] Disabled cudnn.benchmark for structural-loss stability.")
        else:
            torch.backends.cudnn.benchmark = True

    # --- Initialization ---
    # pl.seed_everything(42, workers=True)
    # try:
    #     mp.set_start_method("spawn")
    # except RuntimeError:
    #     pass

    data_module = WireframeDataModule(
        reg_dir=args.reg_dir,
        depth_dir=args.depth_dir,
        batch_size=args.batch_size,
        num_workers=args.workers,
        threshold=args.threshold,
        use_noisy_data=use_noisy_data,
        occlusion_aware_partial_depth=args.occlusion_aware_partial_depth,
    )

    '''model = Trainer(
        decoder_epochs=args.decoder_epochs,
        decoder_lr=args.decoder_lr,
        encoder_lr=args.encoder_lr,
        weight_decay=args.weight_decay,
    )'''

    # Instantiate VAE (latent 4x32x32) and denoising UNet (sketch-conditioned)
    # IMPORTANT: do NOT hard-code a specific CUDA device index here.
    # Lightning/DDP will move modules to the correct per-rank device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load VAE from Lightning checkpoint
    vae_checkpoint_path = args.vae_checkpoint
    if os.path.exists(vae_checkpoint_path):
        print(f"Loading VAE from checkpoint: {vae_checkpoint_path}")
        checkpoint = torch.load(vae_checkpoint_path, map_location="cpu")
        vae = VAE(latent_ch=4)
        # Extract VAE state dict from Lightning module's 'model' attribute
        vae_state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('model.')}
        vae.load_state_dict(vae_state_dict)
        print("✓ VAE weights loaded successfully")
    else:
        print(f"Warning: VAE checkpoint not found at {vae_checkpoint_path}, using random initialization")
        vae = VAE(latent_ch=4)

    # ========================================================================
    # VERIFY VAE LATENT SPACE SIZE (catch configuration errors early!)
    # ========================================================================
    if is_rank0:
        print("\n" + "="*70)
        print("VERIFYING VAE LATENT SPACE CONFIGURATION")
        print("="*70)
        vae = vae.to(device)
        with torch.no_grad():
            test_input = torch.randn(1, 1, 256, 256, device=device)
            test_mu, test_logvar = vae.encoder(test_input)
            latent_shape = test_mu.shape
            print(f"✓ VAE encoder output shape: {latent_shape}")
            print(f"  Latent space: {latent_shape[1]}×{latent_shape[2]}×{latent_shape[3]}")
            
            expected_spatial = 32  # For ch_mult=(1,2,4,4): 256->128->64->32
            if latent_shape[2] != expected_spatial or latent_shape[3] != expected_spatial:
                print(f"❌ ERROR: Expected {expected_spatial}×{expected_spatial} latents, got {latent_shape[2]}×{latent_shape[3]}")
                print(f"   Check VAE ch_mult configuration!")
                raise ValueError(f"VAE latent space mismatch: expected {expected_spatial}×{expected_spatial}, got {latent_shape[2]}×{latent_shape[3]}")
            
            print(f"✓ Latent spatial size matches expected: {expected_spatial}×{expected_spatial}")
        print("="*70 + "\n")
        # Move back to CPU after checks; Lightning will place it later.
        vae = vae.to("cpu")

    latent_stats_path = args.latent_stats_path

    if not os.path.exists(latent_stats_path):
        if is_rank0:
            print("Computing latent statistics from CLEAN training data...")
            print("(Latent stats are always computed from clean data for consistency)")

            # Get training files to compute stats from (always use clean data for stats)
            clean_files = sorted(glob.glob(os.path.join('abc/zip_clean', '*.npz')))

            # Use every 100th sample from clean dataset for stats (10k samples)
            stats_indices = list(range(0, len(clean_files), 100))

            # Create dataset using clean data only (latent stats should be stable)
            stats_dataset = OcclusionAwareSequentialDataset(
                'abc/zip_clean',
                stats_indices,
                transforms=None,
                use_noisy=False,  # Always use clean data for latent stats
                occlusion_aware_partial_depth=args.occlusion_aware_partial_depth,
            )
            stats_dl_kwargs = {}
            stats_num_workers = 4
            if stats_num_workers > 0:
                stats_dl_kwargs['persistent_workers'] = True
                stats_dl_kwargs['prefetch_factor'] = 2
            stats_dataloader = DataLoader(
                stats_dataset,
                batch_size=128,
                shuffle=True,
                num_workers=stats_num_workers,
                pin_memory=True,
                drop_last=True,
                **stats_dl_kwargs,
            )

            vae = vae.to(device)
            mean_c, std_c, example = compute_latent_stats(
                encoder=vae.encoder,
                dataloader=stats_dataloader,
                device=device
            )
            torch.save({'mean': mean_c, 'std': std_c}, latent_stats_path)
            print(f"✓ Latent statistics saved to {latent_stats_path}")
            vae = vae.to("cpu")
        else:
            # Other ranks wait for rank0 to write the stats file.
            timeout_s = 60 * 30
            start = time.time()
            while not os.path.exists(latent_stats_path):
                if time.time() - start > timeout_s:
                    raise TimeoutError(f"Timed out waiting for latent stats at {latent_stats_path}")
                time.sleep(2.0)

    # Build denoiser UNet with ControlNet support
    # Determine control channel mapping if using ControlNet
    if args.use_controlnet:
        # Sketch encoder (VAE encoder with base_ch=32, ch_mult=(1,2,4,4)) produces features at:
        # - 128x128: 64 channels (base_ch * 2)
        # - 64x64: 128 channels (base_ch * 4)
        # - 32x32: 128 channels (base_ch * 4)
        #
        # UNet decoder operates at 32x32, 16x16, 8x8 (init_res=32, downsamples to 8, upsamples back)
        # We map decoder resolutions to control feature channels:
        control_channels = {
            32: 128,  # decoder 32x32 uses control feature from 32x32 (128ch)
            16: 128,  # decoder 16x16 uses control feature from 64x64 downsampled (128ch)
            8: 128,   # decoder 8x8 uses control feature from 64x64 downsampled further (128ch)
        }
        use_controlnet = True
        print("✓ ControlNet mode enabled - spatial feature injection active")
        print(f"✓ ControlNet encoder: {args.control_encoder}")
    else:
        control_channels = None
        use_controlnet = False
        print("✓ Standard mode - concatenation-based conditioning")
    
    # Noise conditioning should ONLY be enabled when training on noisy data
    # Clean data: noise_scalar is always 0.0, so conditioning is unnecessary and harmful
    # Noisy data: noise_scalar varies [0, 1], model needs to learn noise-aware denoising
    use_noise_conditioning = use_noisy_data
    print(f"✓ Noise conditioning: {'ENABLED' if use_noise_conditioning else 'DISABLED'} (matches data mode)")

    unet_channel_mult = _parse_int_tuple(args.unet_channel_mult, "unet_channel_mult")
    unet_attn_resolutions = _parse_int_tuple(args.unet_attn_resolutions, "unet_attn_resolutions")
    if is_rank0:
        print("\n" + "="*70)
        print("UNET CAPACITY CONFIG")
        print("="*70)
        print(f"  model_channels: {args.unet_model_channels}")
        print(f"  num_res_blocks: {args.unet_num_res_blocks}")
        print(f"  channel_mult: {unet_channel_mult}")
        print(f"  attn_resolutions: {unet_attn_resolutions}")
        print("="*70 + "\n")
    
    unet = DenoisingUNet2DConditionModel(
        latent_channels=4,
        model_channels=args.unet_model_channels,
        channel_mult=unet_channel_mult,
        num_res_blocks=args.unet_num_res_blocks,
        attn_resolutions=unet_attn_resolutions,
        num_heads=8,
        dropout=0.0,
        init_res=32,  # Set to match actual latent spatial resolution (32x32)
        use_controlnet=use_controlnet,
        control_channels=control_channels,
        use_noise_conditioning=use_noise_conditioning,
    )

    # ========================================================================
    # VERIFY CONTROLNET CONFIGURATION (if enabled)
    # ========================================================================
    if args.use_controlnet and is_rank0:
        print("\n" + "="*70)
        print("VERIFYING CONTROLNET CONFIGURATION")
        print("="*70)
        with torch.no_grad():
            # Run checks on GPU to match actual kernels, but only on rank0.
            test_sketch = torch.randn(1, 3, 256, 256, device=device)
            if args.control_encoder == "vae":
                # VAE ControlEncoder extracts at 128/64/32 (encoder path resolutions)
                vae = vae.to(device)
                sketch_encoder_test = vae.duplicate_encoder_for_sketch(sketch_in_channels=3).to(device)
                from sketch_recon.models.vae_unet_control import ControlEncoder
                control_encoder_test = ControlEncoder(sketch_encoder_test, extract_resolutions=(128, 64, 32)).to(device)
                expected_control_res = {128, 64, 32}
            elif args.control_encoder == "dinov2":
                # DinoV2 control encoder emits at decoder resolutions (32/16/8)
                control_encoder_test = DinoV2ControlEncoder(
                    model_name=args.dinov2_model,
                    out_channels=128,
                    target_resolutions=(32, 16, 8),
                    train_backbone=args.train_dinov2_backbone,
                ).to(device)
                expected_control_res = {32, 16, 8}
            else:
                # Scratch ViT-small emits at decoder resolutions (32/16/8)
                control_encoder_test = ViTSmallScratchControlEncoder(
                    model_name="vit_small_patch16_224",
                    out_channels=128,
                    target_resolutions=(32, 16, 8),
                    in_channels=3,
                ).to(device)
                expected_control_res = {32, 16, 8}

            control_features_test = control_encoder_test(test_sketch)
            print(f"✓ Control features extracted at resolutions: {sorted(control_features_test.keys())}")
            for res, feat in sorted(control_features_test.items()):
                print(f"  {res}×{res}: {feat.shape}")

            actual_control_res = set(control_features_test.keys())
            if actual_control_res != expected_control_res:
                print(f"❌ ERROR: Expected control resolutions {expected_control_res}, got {actual_control_res}")
                raise ValueError("Control encoder resolution mismatch")
            
            # Test UNet forward pass with control features
            unet = unet.to(device)
            test_zt = torch.randn(1, 4, 32, 32, device=device)
            test_t = torch.tensor([500], device=device)
            test_sketch_latent = torch.randn(1, 4, 32, 32, device=device)
            
            noise_pred = unet(test_zt, test_t, test_sketch_latent, control_features=control_features_test)
            print(f"✓ UNet forward pass successful with ControlNet")
            print(f"  Input: {test_zt.shape} → Output: {noise_pred.shape}")
            
            if noise_pred.shape != test_zt.shape:
                print(f"❌ ERROR: Output shape {noise_pred.shape} doesn't match input {test_zt.shape}")
                raise ValueError("UNet output shape mismatch")
            
            print(f"✓ All ControlNet checks passed!")
        print("="*70 + "\n")
        # Move back to CPU after checks; Lightning will place it later.
        vae = vae.to("cpu")
        unet = unet.to("cpu")

    # ========================================================================
    # VERIFY DATASET CONFIGURATION
    # ========================================================================
    if is_rank0:
        print("\n" + "="*70)
        print("VERIFYING DATASET CONFIGURATION")
        print("="*70)
        
        # Setup data module to check dataset behavior
        data_module.setup()
        
        # Sample a few batches from training dataset
        print("Checking training dataset...")
        train_sample_loader = DataLoader(data_module.train_ds, batch_size=4, shuffle=True, num_workers=0)
        train_batch = next(iter(train_sample_loader))
        imgs, gt_depth, noise_scalars = train_batch
        
        print(f"✓ Training batch shapes:")
        print(f"  Input (sketch): {imgs.shape} - Expected: (B, 3, H, W)")
        print(f"  GT depth: {gt_depth.shape} - Expected: (B, 1, H, W)")
        print(f"  Noise scalars: {len(noise_scalars)} values - Expected: B floats")
        
        if use_noisy_data:
            print(f"\n✓ NOISY DATA MODE ACTIVE")
            print(f"  Noise scalar range (normalized [0,1]): [{min(noise_scalars):.4f}, {max(noise_scalars):.4f}]")
            # Convert back to raw jitter strength for display
            # New format: noise_scalar in [0,1] maps to jitter strength [0, 0.15] (% of min edge length)
            JITTER_MAX = 0.15
            raw_jitter_min = min(noise_scalars) * JITTER_MAX
            raw_jitter_max = max(noise_scalars) * JITTER_MAX
            print(f"  Raw jitter strength: [{raw_jitter_min:.4f}, {raw_jitter_max:.4f}] (range [0, {JITTER_MAX}])")
            
            # Verify noisy sketch mask is being used
            if imgs[:, 0].max() > 0:
                print(f"  ✓ Sketch mask channel has non-zero values (noisy sketch loaded)")
            else:
                print(f"  ⚠️  WARNING: Sketch mask channel is all zeros!")
        else:
            print(f"\n✓ CLEAN DATA MODE ACTIVE")
            print(f"  Noise scalars should all be 0.0: [{min(noise_scalars):.4f}, {max(noise_scalars):.4f}]")
            if max(noise_scalars) > 0.01:
                print(f"  ⚠️  WARNING: Non-zero noise scalars in clean mode!")
        
        # Check value ranges
        print(f"\n✓ Value ranges:")
        print(f"  Input channel 0 (sketch mask): [{imgs[:, 0].min():.3f}, {imgs[:, 0].max():.3f}]")
        print(f"  Input channel 1 (partial depth hint): [{imgs[:, 1].min():.3f}, {imgs[:, 1].max():.3f}]")
        print(f"  Input channel 2 (hint_valid mask): [{imgs[:, 2].min():.3f}, {imgs[:, 2].max():.3f}]")
        print(f"  GT depth: [{gt_depth.min():.3f}, {gt_depth.max():.3f}]")
        
        print("="*70 + "\n")
    else:
        # Other ranks still need datasets initialized for training.
        data_module.setup()
    model = SketchDenoiserModule(
        vae=vae,
        unet=unet,
        latent_stats_path=latent_stats_path,
        img_size=256,
        lr=args.decoder_lr,
        control_lr=args.encoder_lr,
        weight_decay=args.weight_decay,
        snr_weight=True,
        cfg_drop_p=0.15,
        T=1000,
        sketch_in_channels=3,
        use_controlnet=use_controlnet,
        control_encoder_type=args.control_encoder,
        dinov2_model_name=args.dinov2_model,
        train_dinov2_backbone=args.train_dinov2_backbone,
        lambda_grad=args.lambda_grad,
        struct_t_max=args.struct_t_max,
        empty_cache_each_epoch=args.empty_cache_each_epoch,
    )

    if resolved_channels_last:
        try:
            model = model.to(memory_format=torch.channels_last)
        except Exception:
            pass

    if resolved_torch_compile:
        if is_rank0:
            print(f"Compiling UNet with torch.compile(mode={args.torch_compile_mode})...")
        try:
            model.unet = torch.compile(model.unet, mode=args.torch_compile_mode)
        except Exception as e:
            if is_rank0:
                print(f"⚠️  torch.compile failed; continuing without compile. Error: {e}")

    # model = ScratchTrainer(img_size=128, lr=5e-5)

    checkpoint_callback = ModelCheckpoint(
        # dirpath=args.checkpoint_path,
        filename='best-model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        verbose=True,
        monitor='val_diff', # monitor='val_loss',
        mode='min'
    )

    time_checkpoint_callback = ModelCheckpoint(
        # dirpath="checkpoints/",
        filename="time-step-{step}",
        train_time_interval=timedelta(minutes=10),
        save_top_k=0,
        save_last=True
    )

    # Calculate effective batch size for logging
    num_devices = args.devices if isinstance(args.devices, int) else len(args.devices)
    ddp_strategy = "auto"
    if torch.cuda.is_available() and int(num_devices) > 1:
        ddp_strategy = DDPStrategy(
            static_graph=False,
            gradient_as_bucket_view=resolved_gradient_as_bucket_view,
            bucket_cap_mb=25,
        )
    effective_batch_size = args.batch_size * args.accumulate_grad_batches * num_devices
    print(f"\n{'='*70}")
    print(f"BATCH SIZE CONFIGURATION")
    print(f"{'='*70}")
    print(f"  Per-GPU batch size: {args.batch_size}")
    print(f"  Gradient accumulation steps: {args.accumulate_grad_batches}")
    print(f"  Number of GPUs: {num_devices}")
    print(f"  Effective batch size: {effective_batch_size}")
    print(f"{'='*70}\n")
    
    # Create logger with explicit version control
    logger = TensorBoardLogger(
        save_dir="lightning_logs",
        name="",  # Empty string to avoid creating a subdirectory
        version=args.log_version,  # If None, creates new version; if specified, resumes that version
    )
    
    trainer = pl.Trainer(
        devices=args.devices,
        accelerator="gpu",
        strategy=ddp_strategy,
        max_epochs=args.epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[checkpoint_callback], # , time_checkpoint_callback],
        logger=logger,
        log_every_n_steps=args.log_every_n_steps,
        precision=resolved_precision,
    )

    trainer.fit(
        model, 
        datamodule=data_module,
        ckpt_path=args.resume_checkpoint # Lightning handles resuming
    )

if __name__ == '__main__':
    main()