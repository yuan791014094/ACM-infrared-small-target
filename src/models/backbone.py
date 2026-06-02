"""ResNet-20 style backbone with reduced down-sampling (paper Table 1).

Channels per stage: Conv-1 -> 16, Stage-1 -> 16, Stage-2 -> 32, Stage-3 -> 64.
Down-sampling is performed ONLY at the first conv of Stage-2 and Stage-3
(480 -> 240 -> 120), i.e. 4x less down-sampling than a standard ResNet, to
preserve small targets in deep layers (paper Sec. 4.4).
"""

import torch.nn as nn


def conv3x3(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    """Standard ResNet basic block: two 3x3 convs + identity shortcut."""
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out


class ResNet20Backbone(nn.Module):
    """Encoder producing 3 feature maps c1 (16ch), c2 (32ch), c3 (64ch).

    Args:
        in_channels: input image channels (1 for grayscale infrared).
        block_per_stage: b in paper Table 1 (b=3 -> standard ResNet-20).
        channels: per-stage channel widths.
        downsample: 'adjusted' (paper Table 1, total /4 by stride at Stage-2/3)
            or 'regular' (standard ResNet: extra /2 in stem maxpool +/2 at
            Stage-1, i.e. 4x more down-sampling, used for the Table 2 ablation).
    """

    def __init__(self, in_channels=1, block_per_stage=3, channels=(16, 32, 64),
                 downsample="adjusted"):
        super().__init__()
        if downsample not in ("adjusted", "regular"):
            raise ValueError(f"unknown downsample scheme: {downsample}")
        self.downsample = downsample
        c1, c2, c3 = channels
        self.stem = nn.Sequential(
            conv3x3(in_channels, c1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        # 'regular' adds two extra /2 stages (stem pool + Stage-1 stride) so the
        # feature maps are down-sampled 4x more than the 'adjusted' scheme.
        if downsample == "regular":
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            s1 = 2
        else:
            self.pool = nn.Identity()
            s1 = 1
        self.layer1 = self._make_stage(c1, c1, block_per_stage, stride=s1)
        self.layer2 = self._make_stage(c1, c2, block_per_stage, stride=2)
        self.layer3 = self._make_stage(c2, c3, block_per_stage, stride=2)
        self.out_channels = channels

    @staticmethod
    def _make_stage(in_ch, out_ch, num_blocks, stride):
        layers = [BasicBlock(in_ch, out_ch, stride)]
        for _ in range(num_blocks - 1):
            layers.append(BasicBlock(out_ch, out_ch, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.pool(x)
        c1 = self.layer1(x)   # /1 (adjusted) or /2 (regular),  16ch
        c2 = self.layer2(c1)  # one more /2,                     32ch
        c3 = self.layer3(c2)  # one more /2,                     64ch
        return c1, c2, c3
