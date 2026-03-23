"""Control encoder backbones used by diffusion training."""

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Dinov2Model


class DinoV2ControlEncoder(nn.Module):
    """Extract multi-scale spatial control features from a DinoV2 backbone."""

    def __init__(
        self,
        model_name: str = "facebook/dinov2-small",
        out_channels: int = 128,
        target_resolutions: tuple[int, ...] = (32, 16, 8),
        train_backbone: bool = False,
        in_channels: int = 3,
    ):
        super().__init__()
        self.model_name = model_name
        self.target_resolutions = tuple(target_resolutions)
        self.backbone = Dinov2Model.from_pretrained(model_name)
        self.backbone.train(mode=train_backbone)
        for p in self.backbone.parameters():
            p.requires_grad = bool(train_backbone)

        if train_backbone:
            for name, p in self.backbone.named_parameters():
                if "mask_token" in name:
                    p.requires_grad = False

        self.in_proj = nn.Conv2d(int(in_channels), 3, kernel_size=1)

        hidden = int(self.backbone.config.hidden_size)
        self.proj = nn.Conv2d(hidden, out_channels, kernel_size=1)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("_rgb_mean", mean, persistent=False)
        self.register_buffer("_rgb_std", std, persistent=False)

    def forward(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=True)
        x = self.in_proj(x)
        x = (x - self._rgb_mean) / (self._rgb_std + 1e-8)

        out = self.backbone(pixel_values=x)
        tokens = out.last_hidden_state[:, 1:, :]

        b, n, d = tokens.shape
        grid = int(round(n**0.5))
        if grid * grid != n:
            raise ValueError(
                f"DinoV2 produced {n} patch tokens (not a square). "
                "Consider adjusting input size to match patch grid."
            )

        feat = tokens.transpose(1, 2).contiguous().view(b, d, grid, grid)
        feat = self.proj(feat)

        features: dict[int, torch.Tensor] = {}
        for res in self.target_resolutions:
            features[res] = F.interpolate(feat, size=(res, res), mode="bilinear", align_corners=True)
        return features


class ViTSmallScratchControlEncoder(nn.Module):
    """Extract multi-scale spatial control features from a scratch-trained ViT-small."""

    def __init__(
        self,
        model_name: str = "vit_small_patch16_224",
        out_channels: int = 128,
        target_resolutions: tuple[int, ...] = (32, 16, 8),
        in_channels: int = 3,
    ):
        super().__init__()
        self.model_name = model_name
        self.target_resolutions = tuple(target_resolutions)
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
            global_pool="",
            in_chans=int(in_channels),
            img_size=224,
        )

        embed_dim = int(getattr(self.backbone, "embed_dim"))
        self.proj = nn.Conv2d(embed_dim, out_channels, kernel_size=1)

    def _to_spatial_tokens(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.forward_features(x)

        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        if feats.ndim != 3:
            raise ValueError(f"Expected ViT forward_features output (B,N,D), got shape {tuple(feats.shape)}")

        b, n, d = feats.shape
        patch_embed = getattr(self.backbone, "patch_embed", None)
        grid_h = grid_w = int(round(n**0.5))
        if patch_embed is not None and hasattr(patch_embed, "grid_size"):
            gs = patch_embed.grid_size
            if isinstance(gs, tuple) and len(gs) == 2:
                grid_h, grid_w = int(gs[0]), int(gs[1])

        if n == grid_h * grid_w + 1:
            feats = feats[:, 1:, :]
            n = feats.shape[1]
        elif n != grid_h * grid_w:
            grid_h = grid_w = int(round(n**0.5))

        if grid_h * grid_w != n:
            raise ValueError(
                f"ViT-small produced {n} patch tokens (not a rectangular grid). "
                "Consider adjusting input size/patch configuration."
            )

        return feats.transpose(1, 2).contiguous().view(b, d, grid_h, grid_w)

    def forward(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=True)
        feat = self._to_spatial_tokens(x)
        feat = self.proj(feat)

        features: dict[int, torch.Tensor] = {}
        for res in self.target_resolutions:
            features[res] = F.interpolate(feat, size=(res, res), mode="bilinear", align_corners=True)
        return features
