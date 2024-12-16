import torch
import torch.nn as nn
import math
from typing import Optional, List, Tuple
from einops import rearrange, repeat

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        normalized = x / rms
        return normalized * self.weight

def apply_rotary_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)
    
    cos = cos[..., :q.shape[-2], :]
    sin = sin[..., :q.shape[-2], :]
    
    qcos = q * cos
    qsin = rotate_half(q) * sin
    q = qcos + qsin
    
    kcos = k * cos
    ksin = rotate_half(k) * sin
    k = kcos + ksin
    
    return q, k

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs)
    cos_freqs = torch.cos(freqs)
    sin_freqs = torch.sin(freqs)
    return cos_freqs, sin_freqs

class ImageModelConfig:
    def __init__(
        self,
        img_size: int = 64,             # Image size (assumed square)
        patch_size: int = 4,            # Patch size for splitting images
        in_channels: int = 3,           # Input image channels
        embed_dim: int = 768,           # Embedding dimension
        depth: int = 12,                # Number of transformer blocks
        num_heads: int = 12,            # Number of attention heads
        mlp_ratio: float = 4.0,         # MLP hidden dim ratio
        qkv_bias: bool = True,          # Use bias in qkv projections
        dropout: float = 0.1,           # Dropout rate
        attn_dropout: float = 0.1,      # Attention dropout rate
        time_embedding_dim: int = 256,  # Time embedding dimension
    ):
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
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size

class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        # Sinusoidal position embedding with MLP
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )
        
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=timesteps.device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        embeddings = self.mlp(embeddings)
        return embeddings

class MultiHeadAttention(nn.Module):
    def __init__(self, config: ImageModelConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(config.embed_dim, config.embed_dim * 3, bias=config.qkv_bias)
        self.attn_drop = nn.Dropout(config.attn_dropout)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.proj_drop = nn.Dropout(config.dropout)
        
        # Precompute RoPE frequencies
        max_seq_len = config.num_patches + 1  # +1 for cls token
        cos_freqs, sin_freqs = precompute_freqs_cis(self.head_dim, max_seq_len)
        self.register_buffer('cos_freqs', cos_freqs)
        self.register_buffer('sin_freqs', sin_freqs)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply RoPE
        q, k = apply_rotary_emb(q, k, self.cos_freqs, self.sin_freqs)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, config: ImageModelConfig):
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
    def __init__(self, config: ImageModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.embed_dim)
        self.attn = MultiHeadAttention(config)
        self.norm2 = RMSNorm(config.embed_dim)
        self.ff = FeedForward(config)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

class ImageTransformer(nn.Module):
    def __init__(self, config: ImageModelConfig):
        super().__init__()
        self.config = config
        
        # Patch embedding
        self.patch_embed = nn.Sequential(
            nn.Linear(config.patch_dim, config.embed_dim),
            RMSNorm(config.embed_dim)
        )
        
        # Time embedding
        self.time_embed = TimeEmbedding(config.time_embedding_dim)
        self.time_proj = nn.Linear(config.time_embedding_dim, config.embed_dim)
        
        # Position embedding
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
        # Initialize embeddings
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        
        # Initialize transformer blocks
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, RMSNorm):
            nn.init.constant_(m.weight, 1.0)
            
    def get_patch_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, N, D)
        B, C, H, W = x.shape
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                     p1=self.config.patch_size, p2=self.config.patch_size)
        x = self.patch_embed(x)
        return x
        
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        
        # Get patch embeddings
        x = self.get_patch_embeddings(x)  # B, N, D
        
        # Add cls token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embeddings
        x = x + self.pos_embed
        
        # Add time embeddings
        time_embed = self.time_embed(timesteps)  # B, D_time
        time_embed = self.time_proj(time_embed)  # B, D
        x = x + time_embed.unsqueeze(1)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
            
        # Decode patches
        x = self.norm(x)
        x = x[:, 1:]  # Remove cls token
        x = self.decoder(x)
        
        # Reshape back to image
        x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                     h=self.config.img_size//self.config.patch_size,
                     w=self.config.img_size//self.config.patch_size,
                     p1=self.config.patch_size,
                     p2=self.config.patch_size,
                     c=self.config.in_channels)
        
        return x

    def get_model_size(self) -> dict:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "transformer_blocks": self.config.depth,
            "embedding_dim": self.config.embed_dim,
            "attention_heads": self.config.num_heads
        }

def main():
    # Create configuration
    config = ImageModelConfig(
        img_size=64,
        patch_size=4,
        in_channels=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        time_embedding_dim=256
    )
    
    # Create model
    model = ImageTransformer(config)
    
    # Print model information
    model_size = model.get_model_size()
    print("Model Size Information:")
    print(f"Total Parameters: {model_size['total_parameters']:,}")
    print(f"Trainable Parameters: {model_size['trainable_parameters']:,}")
    print(f"Transformer Blocks: {model_size['transformer_blocks']}")
    print(f"Embedding Dimension: {model_size['embedding_dim']}")
    print(f"Attention Heads: {model_size['attention_heads']}")
    
    # Test forward pass
    batch_size = 1
    x = torch.randn(batch_size, config.in_channels, 
                   config.img_size, config.img_size)
    timesteps = torch.randint(0, 1000, (batch_size,))
    
    output = model(x, timesteps)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    main()