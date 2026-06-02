"""ACM-FPN and ACM-U-Net (paper Sec. 4.4, Fig. 6).

The host networks (FPN / U-Net) keep their structure, but the cross-layer
feature fusion (the addition in FPN, the concatenation in U-Net) is replaced
by the proposed ACM module:  Z = G(Y) * X + L(X) * Y.

Both networks expose:
  fusion:     cross-layer modulation scheme (for the Table 2 ablation),
              one of acm/topdown_global/topdown_local/bilocal/biglobal.
  downsample: 'adjusted' (paper, /4 total) or 'regular' (/16, ablation).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import ResNet20Backbone
from .acm import CrossLayerFusion


def _up_to(x, ref):
    """Bilinearly upsample x to the spatial size of ref."""
    return F.interpolate(x, size=ref.shape[-2:], mode="bilinear",
                         align_corners=False)


class ACMFPN(nn.Module):
    """FPN host network with ACM cross-layer fusion.

    Lateral 1x1 convs project all backbone features to a common channel
    dim; the top-down addition is replaced by the configurable fusion module
    (X = lateral low-level, Y = upsampled higher level).
    Prediction is read from the finest (full-resolution) level.
    """

    def __init__(self, in_channels=1, block_per_stage=3,
                 channels=(16, 32, 64), fpn_channels=16, reduction=4,
                 fusion="acm", downsample="adjusted",
                 spatial_gate=False, deep_supervision=False):
        super().__init__()
        self.backbone = ResNet20Backbone(in_channels, block_per_stage,
                                         channels, downsample)
        c1, c2, c3 = channels
        d = fpn_channels
        self.deep_supervision = deep_supervision

        self.lat1 = nn.Conv2d(c1, d, kernel_size=1, bias=False)
        self.lat2 = nn.Conv2d(c2, d, kernel_size=1, bias=False)
        self.lat3 = nn.Conv2d(c3, d, kernel_size=1, bias=False)

        self.fuse2 = CrossLayerFusion(d, fusion, reduction, spatial_gate)
        self.fuse1 = CrossLayerFusion(d, fusion, reduction, spatial_gate)

        self.smooth2 = nn.Sequential(
            nn.Conv2d(d, d, 3, padding=1, bias=False),
            nn.BatchNorm2d(d), nn.ReLU(inplace=True))
        self.smooth1 = nn.Sequential(
            nn.Conv2d(d, d, 3, padding=1, bias=False),
            nn.BatchNorm2d(d), nn.ReLU(inplace=True))

        self.head = nn.Conv2d(d, 1, kernel_size=1)
        # ACM++ #2: auxiliary head on the coarser decoder level (deep supervision)
        self.aux_head = nn.Conv2d(d, 1, kernel_size=1) if deep_supervision else None

    def forward(self, x):
        c1, c2, c3 = self.backbone(x)
        p3 = self.lat3(c3)
        p2 = self.smooth2(self.fuse2(self.lat2(c2), _up_to(p3, c2)))
        p1 = self.smooth1(self.fuse1(self.lat1(c1), _up_to(p2, c1)))
        out = _up_to(self.head(p1), x)  # logits, full resolution, 1 channel
        if self.training and self.aux_head is not None:
            aux = _up_to(self.aux_head(p2), x)
            return [out, aux]
        return out


class _UpFuse(nn.Module):
    """Decoder block for U-Net: project deep feature to skip's channels,
    upsample, fuse with skip via the configurable fusion, then refine."""

    def __init__(self, deep_ch, skip_ch, out_ch, reduction=4, fusion="acm",
                 spatial_gate=False):
        super().__init__()
        self.reduce = nn.Conv2d(deep_ch, skip_ch, kernel_size=1, bias=False)
        self.fuse = CrossLayerFusion(skip_ch, fusion, reduction, spatial_gate)
        self.conv = nn.Sequential(
            nn.Conv2d(skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, deep, skip):
        y = _up_to(self.reduce(deep), skip)  # high-level Y
        z = self.fuse(skip, y)               # X = skip (low-level)
        return self.conv(z)


class ACMUNet(nn.Module):
    """U-Net host network with ACM skip-connection fusion."""

    def __init__(self, in_channels=1, block_per_stage=3,
                 channels=(16, 32, 64), reduction=4,
                 fusion="acm", downsample="adjusted",
                 spatial_gate=False, deep_supervision=False):
        super().__init__()
        self.backbone = ResNet20Backbone(in_channels, block_per_stage,
                                         channels, downsample)
        c1, c2, c3 = channels
        self.deep_supervision = deep_supervision
        self.up2 = _UpFuse(c3, c2, c2, reduction, fusion, spatial_gate)
        self.up1 = _UpFuse(c2, c1, c1, reduction, fusion, spatial_gate)
        self.head = nn.Conv2d(c1, 1, kernel_size=1)
        self.aux_head = nn.Conv2d(c2, 1, kernel_size=1) if deep_supervision else None

    def forward(self, x):
        c1, c2, c3 = self.backbone(x)
        d2 = self.up2(c3, c2)
        d1 = self.up1(d2, c1)
        out = _up_to(self.head(d1), x)
        if self.training and self.aux_head is not None:
            aux = _up_to(self.aux_head(d2), x)
            return [out, aux]
        return out


def build_model(name: str, **kwargs):
    name = name.lower()
    if name in ("acm_fpn", "acmfpn", "fpn"):
        return ACMFPN(**kwargs)
    if name in ("acm_unet", "acm_u_net", "acmunet", "unet"):
        return ACMUNet(**kwargs)
    raise ValueError(f"unknown model: {name}")
