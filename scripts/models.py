"""
models.py
Classifier for WAV2VEC2 embeddings.

Fixes vs old version:
- Removed AudioEncoder (was never used in training — dead code)
- Fixed critical bug: bird_classifier was defined TWICE; second definition
  silently overwrote the first and then tried to instantiate itself → crash
- Replaced ReLU+BatchNorm with GELU+LayerNorm (better for embedding spaces)
- Added residual skip connection between 1024→512 blocks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BirdClassifier(nn.Module):
    """
    MLP head that maps WAV2VEC2_LARGE embeddings (2048-dim) to species logits.

    Architecture:
        2048 → 1024 → 512 (residual) → 256 → num_classes
    """

    def __init__(self, input_dim: int = 2048, num_classes: int = 10):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.4),
        )

        # residual: project input to 512 for skip connection
        self.proj = nn.Linear(1024, 512, bias=False)

        self.block2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
        )

        self.block3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        self.head = nn.Linear(256, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)                       # [B, 1024]
        residual = self.proj(x)                  # [B,  512]
        x = self.block2(x) + residual            # [B,  512]  ← skip
        x = self.block3(x)                       # [B,  256]
        return self.head(x)                      # [B, num_classes]
