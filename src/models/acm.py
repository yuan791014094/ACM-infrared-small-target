"""
Asymmetric Contextual Modulation (ACM) module and ablation variants.

Reproduces the core modules of:
  Dai et al., "Asymmetric Contextual Modulation for Infrared Small Target
  Detection", WACV 2020.

Core fusion (paper Eq. 5):
    Z = G(Y) * X + L(X) * Y
where
  X : low-level feature  (C x H x W)
  Y : high-level feature (C x H x W, already upsampled to match X)
  G(Y) : top-down  GLOBAL channel attention  -> shape C x 1 x 1   (Eq. 2)
  L(X) : bottom-up POINT-WISE channel attention -> shape C x H x W (Eq. 3)

For the modulation-scheme ablation (paper Sec. 5.2, Fig. 7) we expose a
configurable `CrossLayerFusion` supporting:
  - 'topdown_global' : one-directional top-down, global attn  (GAU baseline)
  - 'topdown_local'  : one-directional top-down, point-wise attn  (Fig. 7a)
  - 'bilocal'        : bi-directional, both point-wise            (Fig. 7b)
  - 'biglobal'       : bi-directional, both global                (Fig. 7c)
  - 'acm'            : asymmetric: top-down global + bottom-up local (Eq. 5)
"""

import torch
import torch.nn as nn


class GlobalAttention(nn.Module):
    """Global channel attention, paper Eq. 2 / Fig. 4(a).

    GlobalAvgPool -> FC -> BN -> ReLU -> FC -> BN -> Sigmoid, output C x 1 x 1.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        inter = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, inter, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inter)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(inter, channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = self.pool(x)
        w = self.relu(self.bn1(self.fc1(w)))
        w = self.sigmoid(self.bn2(self.fc2(w)))
        return w  # C x 1 x 1


class PointwiseAttention(nn.Module):
    """Point-wise channel attention, paper Eq. 3 / Fig. 4(b).

    PWConv -> BN -> ReLU -> PWConv -> BN -> Sigmoid, output C x H x W.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        inter = max(channels // reduction, 1)
        self.conv1 = nn.Conv2d(channels, inter, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inter)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inter, channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = self.relu(self.bn1(self.conv1(x)))
        w = self.sigmoid(self.bn2(self.conv2(w)))
        return w  # C x H x W


def _make_attention(kind: str, channels: int, reduction: int):
    if kind == "global":
        return GlobalAttention(channels, reduction)
    if kind == "local":
        return PointwiseAttention(channels, reduction)
    raise ValueError(kind)


class SpatialGate(nn.Module):
    """CBAM-style spatial attention (ACM++ improvement #1).

    Aggregates channel-wise avg & max -> 2 x H x W -> conv -> sigmoid -> 1xHxW,
    used to re-weight the fused feature so tiny-target pixels are emphasized.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        a = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * a


# Modulation-scheme registry: (top_down_attn, bottom_up_attn or None)
_FUSION_SPECS = {
    "acm": ("global", "local"),            # Eq. 5 (proposed)
    "topdown_global": ("global", None),    # GAU baseline, Fig. 4(a)
    "topdown_local": ("local", None),      # Fig. 7(a)
    "bilocal": ("local", "local"),         # Fig. 7(b)
    "biglobal": ("global", "global"),      # Fig. 7(c)
}


class CrossLayerFusion(nn.Module):
    """Configurable cross-layer fusion for the modulation ablation.

    Args:
        channels: channels of both X and Y (assumed equal).
        mode: one of _FUSION_SPECS keys.
        reduction: channel reduction ratio r.
        spatial_gate: if True, append a CBAM spatial gate to Z (ACM++).
    Inputs:
        x: low-level feature  (B, C, H, W)
        y: high-level feature (B, C, H, W)  -- must already match x spatially.
    Returns:
        Z = att_td(Y) * X + [att_bu(X) * Y  if bi-directional else Y]
        (optionally refined by a spatial gate)
    """

    def __init__(self, channels: int, mode: str = "acm", reduction: int = 4,
                 spatial_gate: bool = False):
        super().__init__()
        if mode not in _FUSION_SPECS:
            raise ValueError(f"unknown fusion mode: {mode}")
        self.mode = mode
        td_kind, bu_kind = _FUSION_SPECS[mode]
        self.top_down = _make_attention(td_kind, channels, reduction)
        self.bottom_up = (_make_attention(bu_kind, channels, reduction)
                          if bu_kind is not None else None)
        self.spatial = SpatialGate() if spatial_gate else None

    def forward(self, x, y):
        td = self.top_down(y) * x
        z = td + y if self.bottom_up is None else td + self.bottom_up(x) * y
        if self.spatial is not None:
            z = self.spatial(z)
        return z


class ACM(CrossLayerFusion):
    """Asymmetric Contextual Modulation (paper Eq. 5 / Fig. 5).

    Convenience subclass fixing mode='acm'.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__(channels, mode="acm", reduction=reduction)
