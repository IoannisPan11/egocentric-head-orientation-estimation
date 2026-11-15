"""
model_resnet_transformer.py

The CNN-Transformer regression model used in every experiment and in the final model.

Thesis step : Model architecture (Section 4.2)
Run on      : TU Wien dataLAB cluster (SLURM, NVIDIA A100)

Architecture
- `ResNetBackbone`: ResNet-50 (ImageNet weights) without its classification layer, followed by
  a linear projection from 2048 to d_model features per frame.
- `PositionalEncoding`: fixed sinusoidal positional encoding (maximum length 512).
- `ResNetTransformerRegressor`: encodes every frame of the sequence, prepends a learnable CLS
  token, adds the positional encoding and applies a Transformer encoder (pre-norm, GELU,
  batch_first). The CLS output goes through a regression head
  (LayerNorm -> Linear(d_model, 128) -> GELU -> Dropout -> Linear(128, 4)).

Input and output
- Input : frames of shape (B, T, 3, H, W).
- Output: (B, 4) = [delta_roll, delta_pitch, sin(delta_yaw), cos(delta_yaw)].

The dropout value is used both in the encoder layers and in the regression head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm


# =========================================================
# CNN backbone
# =========================================================
class ResNetBackbone(nn.Module):
    def __init__(self, out_dim: int = 128, pretrained: bool = True):
        super().__init__()

        weights = tvm.ResNet50_Weights.DEFAULT if pretrained else None
        net = tvm.resnet50(weights=weights)

        in_dim = net.fc.in_features  # 2048 for ResNet50
        net.fc = nn.Identity()

        self.net = net
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W)
        returns: (B, D)
        """
        feat = self.net(x)          # (B, 2048)
        feat = self.proj(feat)      # (B, D)
        return feat


# =========================================================
# Positional encoding
# =========================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        pe = pe.unsqueeze(0)  # (1, L, D)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        """
        L = x.shape[1]
        return x + self.pe[:, :L, :]


# =========================================================
# Main model
# =========================================================
class ResNetTransformerRegressor(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        pretrained_backbone: bool = True,
        use_cls: bool = True,
        out_dim: int = 4,   # [delta_roll, delta_pitch, sin(delta_yaw), cos(delta_yaw)]
    ):
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        self.d_model = d_model
        self.use_cls = use_cls

        # CNN backbone
        self.backbone = ResNetBackbone(
            out_dim=d_model,
            pretrained=pretrained_backbone,
        )

        # positional encoding
        self.pos = PositionalEncoding(
            d_model=d_model,
            max_len=512
        )

        # transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=enc_layer,
            num_layers=num_layers,
        )

        # CLS token
        if self.use_cls:
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
            head_in = d_model
        else:
            head_in = d_model

        # regression head
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim),
        )

        self._init_parameters()

    def _init_parameters(self):
        if self.use_cls:
            nn.init.normal_(self.cls, mean=0.0, std=0.02)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        frames: (B, T, 3, H, W)

        output:
            (B, 4)
            [delta_roll, delta_pitch, sin(delta_yaw), cos(delta_yaw)]
        """
        B, T, C, H, W = frames.shape

        # CNN features for each frame
        x = frames.reshape(B * T, C, H, W)
        feat = self.backbone(x)                  # (B*T, D)
        feat = feat.reshape(B, T, self.d_model)    # (B, T, D)

        # CLS token
        if self.use_cls:
            cls = self.cls.expand(B, -1, -1)    # (B, 1, D)
            tok = torch.cat([cls, feat], dim=1) # (B, T+1, D)
        else:
            tok = feat

        # positional encoding
        tok = self.pos(tok)

        # transformer
        z = self.encoder(tok)

        # pooling
        if self.use_cls:
            out = z[:, 0, :]        # CLS
        else:
            out = z.mean(dim=1)     # mean pooling

        # regression head
        y = self.head(out)
        return y