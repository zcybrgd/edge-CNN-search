import torch.nn as nn
from search_space import ArchConfig


def _make_divisible(v: float, divisor: int = 8) -> int:
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, expansion_ratio: int,kernel_size: int, stride: int):
        super().__init__()
        hidden_dim = in_channels * expansion_ratio
        self.use_residual = (stride == 1 and in_channels == out_channels)
        padding = kernel_size // 2
        layers = []
        if expansion_ratio != 1:
            layers += [nn.Conv2d(in_channels, hidden_dim, 1, bias=False),nn.BatchNorm2d(hidden_dim),nn.ReLU6(inplace=True),]
        layers += [nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, padding,groups=hidden_dim, bias=False),nn.BatchNorm2d(hidden_dim),nn.ReLU6(inplace=True),nn.Conv2d(hidden_dim, out_channels, 1, bias=False),nn.BatchNorm2d(out_channels),]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.block(x)
        return self.block(x)


class SlimNetStudent(nn.Module):
    def __init__(self, arch: ArchConfig, num_classes: int):
        super().__init__()
        stem_channels = _make_divisible(arch.stem_channels * arch.width_mult)
        self.stem = nn.Sequential(nn.Conv2d(3, stem_channels, 3, stride=1, padding=1, bias=False),nn.BatchNorm2d(stem_channels),nn.ReLU6(inplace=True),)
        blocks = []
        in_channels = stem_channels
        for stage in arch.stages:
            out_channels = _make_divisible(stage.out_channels * arch.width_mult)
            for block_idx in range(stage.num_blocks):
                stride = stage.stride if block_idx == 0 else 1
                blocks.append(InvertedResidualBlock(in_channels, out_channels, stage.expansion_ratio,stage.kernel_size, stride,))
                in_channels = out_channels
        self.blocks = nn.Sequential(*blocks)
        final_channels = _make_divisible(1280 * max(arch.width_mult, 1.0))
        self.head_conv = nn.Sequential(nn.Conv2d(in_channels, final_channels, 1, bias=False),nn.BatchNorm2d(final_channels),nn.ReLU6(inplace=True),)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(final_channels, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head_conv(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


def build_student(arch: ArchConfig, num_classes: int) -> nn.Module:
    return SlimNetStudent(arch, num_classes)