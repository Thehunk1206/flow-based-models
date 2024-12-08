import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from einops import rearrange, repeat

class ModelConfig:
    def __init__(
        self,
        num_frames: int = 16,
        img_size: Optional[Tuple[int, int]] = None,  # Image size (height, width)
        patch_size: int = 4,             # Patch size for splitting images
        in_channels: int = 3,           # Input image channels
        embed_dim: int = 768,           # Embedding dimension
        depth: int = 12,                # Number of transformer blocks
        num_heads: int = 12,            # Number of attention heads
        mlp_ratio: float = 4.0,         # MLP hidden dim ratio
        qkv_bias: bool = True,          # Use bias in qkv projections
        dropout: float = 0.1,           # Dropout rate
        attn_dropout: float = 0.1,      # Attention dropout rate
        time_embedding_dim: int = 256   # Time embedding dimension
    ):
        self.num_frames = num_frames
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.time_embedding_dim = time_embedding_dim
        
        # Derived attributes
        if img_size:
            self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        else:
            self.num_patches = None  # Set dynamically for image-agnostic support
        self.patch_dim = in_channels * patch_size * patch_size

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / norm * self.weight

class RoPE(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        freqs = torch.arange(half_dim, device=x.device).float()
        inv_freq = 1.0 / (10000 ** (freqs / half_dim))
        pos = torch.arange(x.shape[1], device=x.device).float()
        sinusoid_inp = torch.einsum("i,j->ij", pos, inv_freq)

        precomputed_sin = sinusoid_inp.sin()
        precomputed_cos = sinusoid_inp.cos()

        x1, x2 = x[..., :half_dim], x[..., half_dim:]
        return torch.cat([x1 * precomputed_cos - x2 * precomputed_sin, x1 * precomputed_sin + x2 * precomputed_cos], dim=-1)

class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.num_heads
        head_dim = config.embed_dim // config.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(config.embed_dim, config.embed_dim * 3, bias=config.qkv_bias)
        self.attn_drop = nn.Dropout(config.attn_dropout)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.proj_drop = nn.Dropout(config.dropout)

        self.rope = RoPE(head_dim * config.num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply RoPE to queries and keys
        q, k = self.rope(q), self.rope(k)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.embed_dim, int(config.embed_dim * config.mlp_ratio)),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(int(config.embed_dim * config.mlp_ratio), config.embed_dim),
            nn.Dropout(config.dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.embed_dim)
        self.attn = MultiHeadAttention(config)
        self.norm2 = RMSNorm(config.embed_dim)
        self.ff = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

class VideoTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Patch embedding
        self.patch_embed = nn.Sequential(
            nn.Linear(config.patch_dim, config.embed_dim),
            RMSNorm(config.embed_dim)
        )

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Embedding(config.num_frames, config.time_embedding_dim),
            nn.Linear(config.time_embedding_dim, config.embed_dim)
        )

        # Position embedding
        if config.num_patches is None or config.num_patches <= 0:
            raise ValueError("`config.num_patches` must be set and greater than 0.")
        self.pos_embed = nn.Parameter(torch.zeros(1, config.num_patches + 1, config.embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.depth)
        ])

        # Output heads
        self.norm = RMSNorm(config.embed_dim)
        self.decoder = nn.Sequential(
            nn.Linear(config.embed_dim, config.patch_dim),
            nn.Tanh()  # Normalize output to [-1, 1]
        )

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, RMSNorm):
            nn.init.constant_(m.weight, 1.0)

    def get_patch_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        B, F, C, H, W = x.shape
        h_mod, w_mod = H % self.config.patch_size, W % self.config.patch_size

        if h_mod != 0 or w_mod != 0:
            raise ValueError("Image height and width must be divisible by patch size.")

        x = rearrange(x, 'b f c (h p1) (w p2) -> b f (h w) (p1 p2 c)',
                     p1=self.config.patch_size, p2=self.config.patch_size)
        x = self.patch_embed(x)
        return x

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        B, F, C, H, W = x.shape

         # Validate input dimensions
        h_mod, w_mod = H % self.config.patch_size, W % self.config.patch_size
        if h_mod != 0 or w_mod != 0:
            raise ValueError("Image height and width must be divisible by patch size.")

        # Compute patch embeddings
        x = self.get_patch_embeddings(x)  # Shape: (B, F, num_patches, embed_dim)

        # Combine time and patch embeddings
        x = rearrange(x, 'b f n d -> b (f n) d')  # Flatten frames and patches
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)  # Shape: (B, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # Add cls_token

        # Add positional embeddings
        x = x + self.pos_embed

        # Time embedding
        time_emb = self.time_embed(timesteps)  # Shape: (B, embed_dim)
        time_emb = repeat(time_emb, 'b d -> b n d', n=x.shape[1])  # Shape: (B, seq_len, embed_dim)
        x = x + time_emb

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Normalize and decode
        x = self.norm(x)
        x = self.decoder(x[:, 1:])  # Exclude cls_token

        # Reshape back to the original video format
        x = rearrange(x, 'b (f n) (p1 p2 c) -> b f c (h p1) (w p2)',
                      f=F, n=self.config.num_patches,
                      p1=self.config.patch_size, p2=self.config.patch_size,
                      h=H // self.config.patch_size, w=W // self.config.patch_size)
        return x

if __name__ == "__main__":
    import torch

    # Define the configuration for the model
    config = ModelConfig(
        num_frames=16,
        img_size=(64, 64),  # Example image size
        patch_size=4,
        in_channels=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        dropout=0.1,
        attn_dropout=0.1,
        time_embedding_dim=256
    )

    # Initialize the VideoTransformer model
    model = VideoTransformer(config)
    model.to(device="cuda:4")

    model_state = model.state_dict()

    print(model_state.keys())


    # Create a sample input tensor
    # Shape: (batch_size, num_frames, channels, height, width)
    sample_input = torch.randn(2, 16, 3, 64, 64)  # Example with batch size of 2
    sample_input = sample_input.to(device="cuda:4")

    # Create a sample timesteps tensor
    # Shape: (batch_size, num_frames)
    timesteps = torch.arange(16).repeat(2, 1)  # Example timesteps for each frame
    timesteps = timesteps.to(device="cuda:4")
    # print(timesteps.device)
    # print(sample_input.device)
    # Forward pass through the model
    output = model(sample_input, timesteps)

    # # Print the output shape
    print("Output shape:", output.shape)

