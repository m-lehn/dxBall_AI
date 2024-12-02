import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange
from torch import nn, optim, Tensor

class Head(nn.Module):
    def __init__(self, dim: int, head_size: int):
        """
        One Head of Self Attention containing 3 linear layers to 
        project an input into query, key and value, and perform
        the self attention mechanism.
        """
        super().__init__()
        self.q = nn.Linear(dim, head_size, bias=False) # query
        self.k = nn.Linear(dim, head_size, bias=False) # key
        self.v = nn.Linear(dim, head_size, bias=False) # value

        # if query and key are unit variance, 
        # the scaled dot product will be unit variance too
        self.scale = dim ** -0.5
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Inputs:
            x: Tensor of shape [B, N, C]

        Returns: Tensor of shape [B, head_size, C]
        """
        q = self.q(x)  # [B, N, C]
        k = self.k(x)  # [B, N, C]
        v = self.v(x)  # [B, N, C]
        
        scores = einsum(q, k, 'B N C, B M C -> B N M') * self.scale  # [B, N, N]
        weights = scores.softmax(dim=-1)
        context = einsum(weights, v, 'B N M, B M C -> B N C')
        return context

class MultiHeadAttention(nn.Module):
    """
    Multi Head Attention Module which applies 'heads' times SelfAttention 
    on the input.
    """
    def __init__(self, dim: int, heads: int, dropout: float = 0.2):
        super().__init__()
        assert dim % heads == 0, "dim must be a multiple of heads"
        headsize = dim // heads
        self.heads = nn.ModuleList([Head(dim, headsize) for _ in range(heads)])
        self.proj  = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)  # Regularization
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Inputs:
            x: Tensor of shape [B, N, C]

        Returns: Tensor of shape [B, N, C]
        """
        out = torch.cat([h(x) for h in self.heads], dim=2)
        out = self.proj(out)
        out = self.dropout(out)  # Regularization
        return out

class Block(nn.Module):
    def __init__(self, dim: int, heads: int, ff_dim: int = None, dropout: float = 0.2):
        super().__init__()
        self.attn = MultiHeadAttention(dim, heads, dropout)
        self.ffwd = FeedForward(dim, ff_dim, dropout)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class FeedForward(nn.Module):
    def __init__(self, dim: int, ff_dim: int = None, dropout: float = 0.2):
        super().__init__()
        ff_dim = ff_dim or dim * 4  # Default to 4x hidden dimension
        self.net = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class TransformerEncoder(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, ff_dim: int = None, dropout: float = 0.2):
        super().__init__()
        self.layers = nn.ModuleList([Block(dim, heads, ff_dim, dropout) for _ in range(depth)])
        self.ln = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.ln(x)  # Final layer normalization

class FramePredictor(nn.Module):
    def __init__(self, seq_size=5, img_size=50, patch_size=10, dim=256, depth=4, heads=8):
        super().__init__()
        self.seq_size = seq_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.dim = dim

        # Patch embedding
        self.embedding = nn.Conv2d(seq_size, dim, kernel_size=patch_size, stride=patch_size)

        # Learnable positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, self.num_patches, dim))

        # Transformer encoder
        self.encoder = TransformerEncoder(dim, depth, heads)

        # Output projection
        self.to_image = nn.ConvTranspose2d(dim, 1, kernel_size=patch_size, stride=patch_size)
        #self.to_image = nn.Conv2d(dim, 1, kernel_size=1)

    def forward(self, x):
        """
        x: [batch_size, sequence_length, channels, height, width]
        """
        batch_size, seq_length, channels, height, width = x.shape

        # Combine sequence_length into the channel dimension
        x = x.view(batch_size, seq_length * channels, height, width)  # [batch_size, sequence_length * channels, height, width]

        # Apply Conv2d patch embedding
        x = self.embedding(x)  # [batch_size, dim, num_patches_y, num_patches_x]

        # Flatten patches and prepare for Transformer
        num_patches = x.size(2) * x.size(3)  # Total number of patches
        x = x.flatten(2).transpose(1, 2)  # [batch_size, num_patches, dim]

        # Add positional encoding
        x = x + self.positional_encoding  # [batch_size, num_patches, dim]

        # Pass through Transformer encoder
        x = self.encoder(x)  # [batch_size, num_patches, dim]

        # Reshape and reconstruct patches
        x = x.transpose(1, 2).view(batch_size, self.dim, height // self.patch_size, width // self.patch_size)
        return self.to_image(x)  # [batch_size, 1, height, width]
