# an actual VAE: encode gt depth, reconstruct gt depth ONLY

import cv2
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForDepthEstimation, AutoImageProcessor, AutoConfig, Dinov2Model, AutoModel, get_linear_schedule_with_warmup


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0, groups=32):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.act1 = SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.act2 = SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        h = self.conv1(self.act1(self.norm1(x)))
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h + self.skip(x)


class MultiHeadSelfAttention2D(nn.Module):
    """Memory-efficient multi-head self-attention over spatial tokens (H*W).

    Uses PyTorch scaled_dot_product_attention (Flash / Mem-Efficient kernels when available)
    with a safe fallback to manual softmax attention. Keeps 1x1 conv projections like SD VAE.
    """
    def __init__(self, channels: int, num_heads: int = 8, dropout: float = 0.0, groups: int = 32, use_sdpa: bool = True):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.use_sdpa = use_sdpa

        self.norm = nn.GroupNorm(groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.attn_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.proj_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        t = h * w
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm)
        q, k, v = torch.chunk(qkv, 3, dim=1)  # (b,c,h,w)

        # reshape -> (b, heads, tokens, head_dim)
        def to_heads(tensor: torch.Tensor):
            tensor = tensor.view(b, self.num_heads, self.head_dim, t).permute(0, 1, 3, 2).contiguous()  # b,h,T,d
            return tensor

        qh = to_heads(q)
        kh = to_heads(k)
        vh = to_heads(v)

        # flatten heads for sdpa: (b*h, T, d)
        qf = qh.reshape(b * self.num_heads, t, self.head_dim)
        kf = kh.reshape(b * self.num_heads, t, self.head_dim)
        vf = vh.reshape(b * self.num_heads, t, self.head_dim)

        if self.use_sdpa:
            try:
                with torch.backends.cuda.sdpa_kernel(enable_flash=True, enable_math=True, enable_memory_efficient=True):
                    of = F.scaled_dot_product_attention(qf, kf, vf, dropout_p=0.0, is_causal=False)
            except Exception:
                attn = torch.matmul(qf, kf.transpose(-2, -1).contiguous()) / (self.head_dim ** 0.5)
                attn = F.softmax(attn, dim=-1)
                attn = self.attn_drop(attn)
                of = torch.matmul(attn, vf)
        else:
            attn = torch.matmul(qf, kf.transpose(-2, -1).contiguous()) / (self.head_dim ** 0.5)
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            of = torch.matmul(attn, vf)

        # reshape back: (b*h, T, d) -> (b, c, h, w)
        of = of.view(b, self.num_heads, t, self.head_dim).permute(0, 1, 3, 2).contiguous().reshape(b, c, h, w)
        out = self.proj_out(of)
        out = self.proj_drop(out)
        return x + out


class FeedForward2D(nn.Module):
    """Position-wise MLP implemented with 1x1 convs over 2D feature maps."""
    def __init__(self, channels: int, expansion: float = 4.0, dropout: float = 0.0, groups: int = 32):
        super().__init__()
        hidden = int(channels * expansion)
        self.norm = nn.GroupNorm(groups, channels)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.act = SiLU()
        self.drop1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)
        self.drop2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.fc2(self.drop1(self.act(self.fc1(h))))
        h = self.drop2(h)
        return x + h


class TransformerBlock2D(nn.Module):
    """Transformer block for 2D feature maps: MHA + MLP with residuals."""
    def __init__(self, channels: int, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.0, groups: int = 32):
        super().__init__()
        self.attn = MultiHeadSelfAttention2D(channels, num_heads=num_heads, dropout=dropout, groups=groups)
        self.mlp = FeedForward2D(channels, expansion=mlp_ratio, dropout=dropout, groups=groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x)
        x = self.mlp(x)
        return x


class Encoder(nn.Module):
    """Depth-only encoder: 256x256x1 -> 16x16x(2*latent_ch) for mu and logvar."""
    def __init__(
        self,
        in_ch=1,
        base_ch=32,
        ch_mult=(1, 2, 4, 4),
        num_res_blocks=1,
        attn_resolutions=(16,),
        latent_ch=4,
        dropout=0.0,
        attn_heads: int = 8,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.attn_resolutions = set(attn_resolutions)
        self.init = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        in_channels = base_ch
        curr_res = 256
        blocks = []
        chs = []
        for i, mult in enumerate(ch_mult):
            out_channels = base_ch * mult
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(in_channels, out_channels, dropout=dropout))
                in_channels = out_channels
                # if curr_res in self.attn_resolutions:
                #     blocks.append(TransformerBlock2D(in_channels, num_heads=attn_heads, mlp_ratio=mlp_ratio, dropout=dropout))
            chs.append(in_channels)
            if i != len(ch_mult) - 1:
                blocks.append(nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1))
                curr_res //= 2
        self.blocks = nn.ModuleList(blocks)

        self.norm_out = nn.GroupNorm(32, in_channels)
        self.act = SiLU()
        self.mu_logvar = nn.Conv2d(in_channels, latent_ch * 2, 3, padding=1)

    def forward(self, x):
        h = self.init(x)
        curr_res = 256
        for m in self.blocks:
            h = m(h)
            if isinstance(m, nn.Conv2d) and m.stride == (2, 2):
                curr_res //= 2
        h = self.mu_logvar(self.act(self.norm_out(h)))
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar


class Decoder(nn.Module):
    """Decoder: 16x16xlatent_ch -> 256x256x1."""
    def __init__(
        self,
        out_ch=1,
        base_ch=32,
        ch_mult=(4, 4, 2, 1),
        num_res_blocks=2,
        attn_resolutions=(16,),
        latent_ch=4,
        dropout=0.0,
        attn_heads: int = 8,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.attn_resolutions = set(attn_resolutions)

        in_channels = base_ch * ch_mult[0]
        self.z_proj = nn.Conv2d(latent_ch, in_channels, 3, padding=1)

        blocks = []
        curr_res = 16
        for i, mult in enumerate(ch_mult):
            out_channels = base_ch * mult
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(in_channels, out_channels, dropout=dropout))
                in_channels = out_channels
                # if curr_res in self.attn_resolutions:
                #     blocks.append(TransformerBlock2D(in_channels, num_heads=attn_heads, mlp_ratio=mlp_ratio, dropout=dropout))
            if i != len(ch_mult) - 1:
                blocks.append(nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1))
                curr_res *= 2
        self.blocks = nn.ModuleList(blocks)

        self.norm_out = nn.GroupNorm(32, in_channels)
        self.act = SiLU()
        self.final = nn.Conv2d(in_channels, out_ch, 3, padding=1)

    def forward(self, z):
        h = self.z_proj(z)
        curr_res = 16
        for m in self.blocks:
            h = m(h)
            if isinstance(m, nn.ConvTranspose2d) and m.stride == (2, 2):
                curr_res *= 2
        h = self.final(self.act(self.norm_out(h)))
        return h


class VAE(nn.Module):
    """
    A simple VAE that takes only GT depth as input:
    - Encoder compresses 256x256x1 -> 16x16x(2*latent_ch) for mu/logvar
    - Decoder reconstructs depth from sampled z (16x16xlatent_ch)
    - Uses ResNet blocks and a spatial attention block at 32x32 resolution
    """
    def __init__(self, latent_ch=4, base_ch=32, num_res_blocks=1, dropout=0.0, attn_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        # Attention at 32x32 like SD VAE
        attn_res = (16,)
        self.latent_ch = latent_ch
        self.base_ch = base_ch
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout
        self.attn_heads = attn_heads
        self.mlp_ratio = mlp_ratio
        self.attn_res = attn_res
        
        self.encoder = Encoder(
            in_ch=1,
            base_ch=base_ch,
            ch_mult=(1, 2, 4, 4),  # 256->128->64->32->16
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_res,
            latent_ch=latent_ch,
            dropout=dropout,
            attn_heads=attn_heads,
            mlp_ratio=mlp_ratio,
        )
        self.decoder = Decoder(
            out_ch=1,
            base_ch=base_ch,
            ch_mult=(4, 4, 2, 1),  # 16->32->64->128->256
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_res,
            latent_ch=latent_ch,
            dropout=dropout,
            attn_heads=attn_heads,
            mlp_ratio=mlp_ratio,
        )

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, depth, sample=True):
        mu, logvar = self.encoder(depth)
        z = self.reparameterize(mu, logvar) if sample else mu
        recon = self.decoder(z)
        return recon, mu, logvar, z

    def duplicate_encoder_for_sketch(self, sketch_in_channels: int = 2) -> Encoder:
        """Create a duplicate encoder for sketch input with adapted first conv layer.
        
        Args:
            sketch_in_channels: Number of input channels for sketch (default: 2)
            
        Returns:
            A new Encoder instance with duplicated weights, first conv adapted to sketch_in_channels
        """
        # Create new encoder with same architecture but different input channels
        sketch_encoder = Encoder(
            in_ch=sketch_in_channels,
            base_ch=self.base_ch,
            ch_mult=(1, 2, 4, 4),
            num_res_blocks=self.num_res_blocks,
            attn_resolutions=self.attn_res,
            latent_ch=self.latent_ch,
            dropout=self.dropout,
            attn_heads=self.attn_heads,
            mlp_ratio=self.mlp_ratio,
        )
        
        # Get original encoder state dict and remove the first conv layer
        original_state_dict = self.encoder.state_dict()
        init_weight_key = 'init.weight'
        init_bias_key = 'init.bias'
        
        # Remove init layer from state dict
        state_dict_to_load = {k: v for k, v in original_state_dict.items() 
                              if not k.startswith('init.')}
        
        # Load all weights except the first conv layer
        sketch_encoder.load_state_dict(state_dict_to_load, strict=False)
        
        # Adapt the first conv layer: duplicate 1-channel weights to sketch_in_channels and divide by sketch_in_channels
        original_weight = original_state_dict[init_weight_key]  # shape: (out_ch, 1, 3, 3)
        new_weight = original_weight.repeat(1, sketch_in_channels, 1, 1) / sketch_in_channels  # (out_ch, sketch_in_channels, 3, 3)
        sketch_encoder.init.weight.data = new_weight
        
        # Copy bias if it exists
        if init_bias_key in original_state_dict:
            sketch_encoder.init.bias.data = original_state_dict[init_bias_key].clone()
        
        return sketch_encoder


# ------------------------------
# Sketch-conditioned denoising UNet ( )n
# ------------------------------


def sinusoidal_timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Create sinusoidal timestep embeddings like in DDPM/SD.

    Args:
        t: (B,) tensor of timesteps (int or float)
        dim: embedding dimension (even)
    Returns:
        (B, dim) tensor
    """
    half = dim // 2
    device = t.device
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=device).float() / half)
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


class TimeEmbedding(nn.Module):
    """Sinusoidal embedding followed by small MLP."""
    def __init__(self, model_channels: int, time_embed_mult: int = 4):
        super().__init__()
        time_dim = model_channels * time_embed_mult
        self.time_dim = time_dim
        self.proj = nn.Sequential(
            nn.Linear(model_channels, time_dim),
            SiLU(),
            nn.Linear(time_dim, time_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # map t -> (B, model_channels) using sinusoidal embedding, then MLP -> (B, time_dim)
        base = sinusoidal_timestep_embedding(t, self.proj[0].in_features)
        return self.proj(base)


class NoiseScalarEmbedding(nn.Module):
    """MLP to project noise scalar (float) to time embedding dimension.
    
    Takes a single normalized float scalar in [0, 1] indicating noise level:
    - 0.0 = minimum noise (0.02 jitter/perlin strength)
    - 1.0 = maximum noise (0.08 jitter/perlin strength)
    
    Projects to the same dimension as time embedding for additive conditioning.
    """
    def __init__(self, time_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(1, time_dim),
            SiLU(),
            nn.Linear(time_dim, time_dim),
        )
    
    def forward(self, noise_scalar: torch.Tensor) -> torch.Tensor:
        """
        Args:
            noise_scalar: (B,) tensor of normalized noise scalars in [0.0, 1.0] range
        Returns:
            (B, time_dim) tensor to be added to time embedding
        """
        # Reshape (B,) -> (B, 1) for linear layer
        x = noise_scalar.view(-1, 1)
        return self.proj(x)


class TimeResBlock(nn.Module):
    """Residual block with time embedding injection (FiLM via bias add)."""
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.0, groups: int = 32):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.act = SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.time_proj = nn.Linear(time_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.dropout(self.act(self.norm2(h))))
        return h + self.skip(x)


class SpatialSelfAttention2D(nn.Module):
    """Multi-head self-attention operating on spatial tokens (H*W)."""
    def __init__(self, channels: int, heads: int = 8, dropout: float = 0.0, groups: int = 32):
        super().__init__()
        assert channels % heads == 0, "channels must be divisible by heads"
        self.heads = heads
        self.head_dim = channels // heads
        self.scale = self.head_dim ** -0.5

        self.norm = nn.GroupNorm(groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.out_proj = nn.Conv2d(channels, channels, 1)
        self.attn_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.proj_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_n = self.norm(x)
        qkv = self.qkv(x_n)
        q, k, v = torch.chunk(qkv, 3, dim=1)  # (b,c,h,w)

        def to_heads(t: torch.Tensor) -> torch.Tensor:
            t = t.reshape(b, self.heads, self.head_dim, h * w)
            return t.permute(0, 1, 3, 2).contiguous()  # (b, heads, tokens, head_dim)
        
        q = to_heads(q).reshape(b * self.heads, h * w, self.head_dim)
        k = to_heads(k).reshape(b * self.heads, h * w, self.head_dim)
        v = to_heads(v).reshape(b * self.heads, h * w, self.head_dim)

        try:
            with torch.backends.cuda.sdpa_kernel(enable_flash=True, enable_math=True, enable_memory_efficient=True):
                out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        except Exception:
            attn = torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.scale
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            out = torch.matmul(attn, v)

        # Reshape 'out' back to (b, c, h, w)
        out = out.view(b, self.heads, h * w, self.head_dim).permute(0, 1, 3, 2).contiguous().reshape(b, c, h, w)
        out = self.out_proj(out)
        out = self.proj_drop(out)
        return x + out


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        # Prefer nearest-neighbor + conv to avoid checkerboard artifacts
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        return self.conv(x)


class DenoisingUNet2DConditionModel(nn.Module):
    """UNet that denoises latent z_t conditioned on sketch latent features via concatenation.

    - Operates on latent space (e.g., VAE latent) with channels=latent_channels and spatial HxW
    - Uses time embedding injected into ResBlocks
    - Adds self-attention at selected resolutions
    - Conditioning is done by concatenating sketch latent (4 channels) with noisy latent (4 channels)
      to create an 8-channel input
    - **ControlNet Enhancement**: Accepts multi-scale control features for spatial injection
    - **Noise Conditioning**: Optional additive noise scalar conditioning (enabled by default)
    """
    def __init__(
        self,
        latent_channels: int = 4,
        model_channels: int = 64,
        channel_mult: tuple = (1, 2, 4),
        num_res_blocks: int = 2,
        attn_resolutions: tuple = (8, 16),
        num_heads: int = 8,
        dropout: float = 0.0,
        groups: int = 32,
        time_embed_mult: int = 4,
        init_res: int = 16,
        use_controlnet: bool = False,
        control_channels: dict = None,  # {res: channels} e.g., {64: 64, 32: 128, 16: 128}
        use_noise_conditioning: bool = True,  # Enable/disable noise scalar conditioning
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.model_channels = model_channels
        self.attn_resolutions = set(attn_resolutions)
        self.num_heads = num_heads
        self.groups = groups
        self.use_controlnet = use_controlnet
        self.init_res = init_res
        self.use_noise_conditioning = use_noise_conditioning

        # Time embedding
        self.time_embed = TimeEmbedding(model_channels, time_embed_mult=time_embed_mult)
        
        # Noise scalar embedding (for conditioning on noise level)
        if self.use_noise_conditioning:
            self.noise_embed = NoiseScalarEmbedding(self.time_embed.time_dim)
        else:
            self.noise_embed = None

        # Input/output projections
        # ControlNet mode: Only noisy latent (4ch), conditioning via spatial features
        # Standard mode: Concatenated latent + sketch latent (8ch)
        if use_controlnet:
            input_channels = latent_channels  # 4ch - no concatenation
        else:
            input_channels = latent_channels * 2  # 8ch - concatenation
        self.in_conv = nn.Conv2d(input_channels, model_channels, 3, padding=1)
        self.out_norm = nn.GroupNorm(groups, model_channels)
        self.out_act = SiLU()
        self.out_conv = nn.Conv2d(model_channels, latent_channels, 3, padding=1)

        # Build UNet with proper skip connection tracking
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        
        ch = model_channels
        curr_res = init_res
        
        # Track skip connection channels for up path
        skip_chs = []
        
        # Build down path: each stage has resblocks + optional attention + optional downsample
        for i, mult in enumerate(channel_mult):
            stage_blocks = nn.ModuleList()
            out_ch = model_channels * mult
            
            for j in range(num_res_blocks):
                stage_blocks.append(TimeResBlock(ch, out_ch, self.time_embed.time_dim, dropout=dropout, groups=groups))
                ch = out_ch
                skip_chs.append(ch)  # Save skip channel count
                if curr_res in self.attn_resolutions:
                    stage_blocks.append(SpatialSelfAttention2D(ch, heads=num_heads, dropout=dropout, groups=groups))
            
            # Add downsample at end of stage (except last stage)
            if i != len(channel_mult) - 1:
                stage_blocks.append(Downsample(ch))
            
            self.downs.append(stage_blocks)
            if i != len(channel_mult) - 1:
                curr_res //= 2

        # Middle blocks
        self.mid_block1 = TimeResBlock(ch, ch, self.time_embed.time_dim, dropout=dropout, groups=groups)
        self.mid_attn = SpatialSelfAttention2D(ch, heads=num_heads, dropout=dropout, groups=groups) if curr_res in self.attn_resolutions else None
        self.mid_block2 = TimeResBlock(ch, ch, self.time_embed.time_dim, dropout=dropout, groups=groups)

        # Build up path: mirror of down path with skip connections
        # Also build zero convs for ControlNet injection at each decoder stage
        self.zero_convs = nn.ModuleDict() if use_controlnet else None
        up_curr_res = curr_res
        
        for i, mult in reversed(list(enumerate(channel_mult))):
            stage_blocks = nn.ModuleList()
            out_ch = model_channels * mult
            
            # num_res_blocks + 1 blocks per stage in up path
            for j in range(num_res_blocks + 1):
                if j < num_res_blocks and skip_chs:
                    # These blocks receive skip connections
                    skip_ch = skip_chs.pop()
                    stage_blocks.append(TimeResBlock(ch + skip_ch, out_ch, self.time_embed.time_dim, dropout=dropout, groups=groups))
                else:
                    # Last block in stage or no more skips
                    stage_blocks.append(TimeResBlock(ch, out_ch, self.time_embed.time_dim, dropout=dropout, groups=groups))
                ch = out_ch
                if curr_res in self.attn_resolutions:
                    stage_blocks.append(SpatialSelfAttention2D(ch, heads=num_heads, dropout=dropout, groups=groups))
            
            # Add upsample at end of stage (except last/top stage)
            if i != 0:
                stage_blocks.append(Upsample(ch))
            
            self.ups.append(stage_blocks)
            
            # Add zero conv for this decoder stage if using ControlNet
            if use_controlnet and control_channels is not None:
                # Map decoder resolution to control feature channels
                # We need to handle resolution mapping: control features might be at 64,32,16
                # but decoder might be at 16,8,4. We'll add zero convs for all decoder resolutions.
                if up_curr_res in control_channels:
                    self.zero_convs[f'res_{up_curr_res}'] = ZeroConv(control_channels[up_curr_res], out_ch)
            
            if i != 0:
                up_curr_res *= 2

    def forward(self, zt: torch.Tensor, t: torch.Tensor, sketch_latent: torch.Tensor = None, 
                control_features: dict = None, noise_scalar: torch.Tensor = None) -> torch.Tensor:
        """Predict v-prediction given noisy latent z_t, timestep t, and sketch conditioning.

        Args:
            zt: (B, latent_channels, H, W) - e.g., (B, 4, 32, 32)
            t: (B,) integer or float timesteps
            sketch_latent: (B, latent_channels, H, W) - encoded sketch latent (for concatenation mode only)
            control_features: Optional dict {resolution: feature_tensor} for ControlNet injection
            noise_scalar: (B,) tensor of noise scalars (floats) indicating sketch noise level
        Returns:
            v_hat: (B, latent_channels, H, W) - v-prediction
        """
        # Compute time embedding
        t_emb = self.time_embed(t)
        
        # Add noise scalar conditioning if enabled and provided
        if self.use_noise_conditioning and noise_scalar is not None and self.noise_embed is not None:
            noise_emb = self.noise_embed(noise_scalar)
            t_emb = t_emb + noise_emb  # Additive conditioning: Global_Embed = Time_Embed + Noise_Embed
        
        # ControlNet mode: No concatenation, conditioning via spatial features only
        # Standard mode: Concatenate noisy latent with sketch latent
        if self.use_controlnet:
            h = self.in_conv(zt)
        else:
            x_in = torch.cat([zt, sketch_latent], dim=1)
            h = self.in_conv(x_in)

        # Down path: collect skip connections after each TimeResBlock
        skips = []
        curr_res = self.init_res
        for stage in self.downs:
            for block in stage:
                if isinstance(block, TimeResBlock):
                    h = block(h, t_emb)
                    skips.append(h)  # Save after every TimeResBlock
                elif isinstance(block, SpatialSelfAttention2D):
                    h = block(h)
                    skips[-1] = h  # Update last skip with attention applied
                elif isinstance(block, Downsample):
                    h = block(h)
                    curr_res //= 2
                    # Don't save downsample output as skip

        # Middle blocks
        h = self.mid_block1(h, t_emb)
        if self.mid_attn is not None:
            h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # Up path: consume skips in reverse order + inject control features
        up_curr_res = curr_res
        for stage_idx, stage in enumerate(self.ups):
            for block in stage:
                if isinstance(block, TimeResBlock):
                    # Concatenate with skip if block was built to expect it (in_ch > out_ch means it expects skip)
                    if block.in_ch != block.out_ch and skips:
                        skip = skips.pop()
                        h = torch.cat([h, skip], dim=1)
                    h = block(h, t_emb)
                    
                    # ControlNet injection: add control features via zero conv
                    if self.use_controlnet and control_features is not None:
                        zero_conv_key = f'res_{up_curr_res}'
                        if zero_conv_key in self.zero_convs:
                            # Find the best matching control feature resolution
                            # Prefer exact match first, then closest resolution
                            available_resolutions = list(control_features.keys())
                            control_res = None
                            
                            # First try exact match
                            if up_curr_res in available_resolutions:
                                control_res = up_curr_res
                            else:
                                # Find closest resolution (prefer larger for better quality when downsampling)
                                sorted_res = sorted(available_resolutions)
                                # Find smallest resolution >= decoder resolution
                                for res in sorted_res:
                                    if res >= up_curr_res:
                                        control_res = res
                                        break
                                # If all control features are smaller, use the largest
                                if control_res is None and sorted_res:
                                    control_res = sorted_res[-1]
                            
                            if control_res is not None:
                                control_feat = control_features[control_res]
                                # Resize control feature to match current spatial resolution
                                if control_feat.shape[2:] != h.shape[2:]:
                                    # Warning: frequent interpolation may indicate misconfiguration
                                    if control_res != up_curr_res and abs(control_res - up_curr_res) > up_curr_res:
                                        # Only warn if we're interpolating from much different resolution
                                        import warnings
                                        warnings.warn(
                                            f"ControlNet: Interpolating control feature from {control_res}×{control_res} "
                                            f"to decoder {up_curr_res}×{up_curr_res}. Consider extracting at decoder resolutions "
                                            f"for better performance.",
                                            UserWarning,
                                            stacklevel=2
                                        )
                                    control_feat = F.interpolate(control_feat, size=h.shape[2:], 
                                                                mode='bilinear', align_corners=True)
                                control_injection = self.zero_convs[zero_conv_key](control_feat)
                                h = h + control_injection
                            
                elif isinstance(block, SpatialSelfAttention2D):
                    h = block(h)
                elif isinstance(block, Upsample):
                    h = block(h)
                    up_curr_res *= 2

        out = self.out_conv(self.out_act(self.out_norm(h)))
        return out


class ZeroConv(nn.Module):
    """Zero-initialized 1x1 convolution for stable ControlNet-style injection.
    
    Starts with zero weights and zero bias, allowing gradual fade-in of control signals
    without disrupting pre-trained features during early training.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        # Zero initialization
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ControlEncoder(nn.Module):
    """Multi-scale feature extractor for ControlNet-style spatial conditioning.
    
    Wraps a pre-trained VAE encoder and extracts intermediate features at multiple
    resolutions. Default (128, 64, 32) is for 32×32 latent space VAE (ch_mult=(1,2,4,4)).
    """
    def __init__(self, encoder: nn.Module, extract_resolutions: tuple = (128, 64, 32)):
        super().__init__()
        self.encoder = encoder
        self.extract_resolutions = set(extract_resolutions)
        
        # We'll extract features after ResBlocks at specified resolutions
        # Features will be stored in a dict during forward pass
        
    def forward(self, x: torch.Tensor) -> dict:
        """Extract multi-scale features from sketch input.
        
        Args:
            x: (B, C, H, W) sketch input at 256x256
            
        Returns:
            dict mapping resolution -> feature tensor.
            For default (128, 64, 32):
            {
                128: (B, C1, 128, 128),
                64: (B, C2, 64, 64),
                32: (B, C3, 32, 32)
            }
        """
        features = {}
        h = self.encoder.init(x)
        curr_res = 256
        
        for m in self.encoder.blocks:
            h = m(h)
            # Track resolution changes
            if isinstance(m, nn.Conv2d) and m.stride == (2, 2):
                curr_res //= 2
            # Extract feature after ResBlock at target resolution
            if isinstance(m, ResBlock) and curr_res in self.extract_resolutions:
                features[curr_res] = h
                
        return features


