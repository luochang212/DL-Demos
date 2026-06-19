"""Fourier Feature Networks — core model classes.

Extracted from the Jupyter notebooks in the parent directory.
Reference: Tancik et al. 2020 (arXiv:2006.10739).
"""

import torch
import torch.nn as nn
from einops import rearrange


class MLP(nn.Module):
    """Pointwise MLP implemented with 1×1 convolutions for 2D feature maps."""

    def __init__(self, in_c: int, out_c: int = 3, hidden_states: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_c, hidden_states, 1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_states),
            nn.Conv2d(hidden_states, hidden_states, 1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_states),
            nn.Conv2d(hidden_states, hidden_states, 1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_states),
            nn.Conv2d(hidden_states, out_c, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class FourierFeature(nn.Module):
    """Random Fourier feature mapping γ(v) = [cos(2πBv), sin(2πBv)].

    B ∈ R^{in_c × (out_c//2)} is drawn from N(0, scale²) and frozen.
    """

    def __init__(self, in_c: int, out_c: int, scale: float):
        super().__init__()
        fourier_basis = torch.randn(in_c, out_c // 2) * scale
        self.register_buffer('_fourier_basis', fourier_basis)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape
        x = rearrange(x, 'n c h w -> (n h w) c')
        x = x @ self._fourier_basis
        x = rearrange(x, '(n h w) c -> n c h w', h=H, w=W)

        x = 2 * torch.pi * x
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=1)
        return x
