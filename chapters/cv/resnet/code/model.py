"""ResNet PyTorch implementation for the DL-Demos tutorial.

Reference:
  He, K., Zhang, X., Ren, S., & Sun, J. (2016).
  Deep Residual Learning for Image Recognition. https://arxiv.org/abs/1512.03385

Provides BasicBlock (ResNet18/34), Bottleneck (ResNet50/101/152),
and factory functions with optional torchvision pretrained weight loading.
"""

from typing import List, Optional, Type, Union

import torch
import torch.nn as nn

__all__ = [
    'BasicBlock',
    'Bottleneck',
    'ResNet',
    'resnet18',
    'resnet34',
    'resnet50',
    'resnet101',
    'resnet152',
]


def conv3x3(in_channels: int, out_channels: int, stride: int = 1):
    """3x3 convolution with padding 1, no bias (bias is in BatchNorm)."""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1(in_channels: int, out_channels: int, stride: int = 1):
    """1x1 convolution, no bias."""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class BasicBlock(nn.Module):
    """Basic residual block used in ResNet-18 and ResNet-34.

    Two 3x3 conv layers. Shortcut is identity when dimensions match,
    otherwise a 1x1 conv for channel/downsample alignment.
    """

    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # Auto-create downsample when dimensions don't match
        if downsample is None and (stride != 1 or in_channels != out_channels):
            downsample = nn.Sequential(
                conv1x1(in_channels, out_channels, stride),
                nn.BatchNorm2d(out_channels),
            )
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    """Bottleneck residual block used in ResNet-50, ResNet-101, and ResNet-152.

    1x1 (reduce) → 3x3 → 1x1 (expand) with expansion factor 4.
    More efficient for deep networks (fewer FLOPs per block vs BasicBlock
    at the same channel depth).
    """

    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        mid_channels = out_channels
        self.conv1 = conv1x1(in_channels, mid_channels)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = conv3x3(mid_channels, mid_channels, stride)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = conv1x1(mid_channels, out_channels * self.expansion)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        # Auto-create downsample when dimensions don't match
        out_dim = out_channels * self.expansion
        if downsample is None and (stride != 1 or in_channels != out_dim):
            downsample = nn.Sequential(
                conv1x1(in_channels, out_dim, stride),
                nn.BatchNorm2d(out_dim),
            )
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """Configurable ResNet backbone.

    Args:
        block: BasicBlock (18/34) or Bottleneck (50/101/152).
        layers: Number of blocks per stage, e.g. [2, 2, 2, 2] for ResNet-18.
        num_classes: Output classes (default 1000 for ImageNet).
    """

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
    ):
        super().__init__()
        self.in_channels = 64

        # Stem: 7x7 conv → BN → ReLU → MaxPool
        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Four stages of residual blocks
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._init_weights()

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        out_channels: int,
        blocks: int,
        stride: int,
    ) -> nn.Sequential:
        """Build a stacked layer of ``blocks`` residual blocks."""
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_channels, out_channels * block.expansion, stride),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _init_weights(self):
        """Kaiming normal initialization for Conv2d and Linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    num_classes: int = 1000,
    pretrained: bool = False,
    **kwargs,
) -> ResNet:
    """Internal factory with optional torchvision pretrained weight loading."""
    model = ResNet(block, layers, num_classes=num_classes, **kwargs)
    if pretrained:
        _load_torchvision_weights(model, block.__name__, layers, num_classes)
    return model


def _load_torchvision_weights(
    model: ResNet,
    block_name: str,
    layers: List[int],
    num_classes: int,
):
    """Load pretrained weights from torchvision, mapping to our ResNet."""
    from torchvision.models import (
        ResNet18_Weights,
        ResNet34_Weights,
        ResNet50_Weights,
        ResNet101_Weights,
        ResNet152_Weights,
    )
    from torchvision.models import (
        resnet18 as tv_resnet18,
    )
    from torchvision.models import (
        resnet34 as tv_resnet34,
    )
    from torchvision.models import (
        resnet50 as tv_resnet50,
    )
    from torchvision.models import (
        resnet101 as tv_resnet101,
    )
    from torchvision.models import (
        resnet152 as tv_resnet152,
    )

    _TV_FACTORIES = {
        ('BasicBlock', (2, 2, 2, 2)): (tv_resnet18, ResNet18_Weights.DEFAULT),
        ('BasicBlock', (3, 4, 6, 3)): (tv_resnet34, ResNet34_Weights.DEFAULT),
        ('Bottleneck', (3, 4, 6, 3)): (tv_resnet50, ResNet50_Weights.DEFAULT),
        ('Bottleneck', (3, 4, 23, 3)): (tv_resnet101, ResNet101_Weights.DEFAULT),
        ('Bottleneck', (3, 8, 36, 3)): (tv_resnet152, ResNet152_Weights.DEFAULT),
    }

    key = (block_name, tuple(layers))
    if key not in _TV_FACTORIES:
        raise ValueError(
            f'No pretrained weights for block={block_name} layers={layers}'
        )

    factory, weights = _TV_FACTORIES[key]
    tv_model = factory(weights=weights)
    state_dict = tv_model.state_dict()

    # Reshape fc weights if num_classes differs from ImageNet 1000
    if num_classes != 1000:
        state_dict.pop('fc.weight', None)
        state_dict.pop('fc.bias', None)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f'[pretrained] missing keys (ok if mismatched num_classes): {missing}')
    if unexpected:
        raise RuntimeError(f'[pretrained] unexpected keys: {unexpected}')


def resnet18(num_classes: int = 1000, pretrained: bool = False, **kwargs) -> ResNet:
    """ResNet-18: BasicBlock × [2, 2, 2, 2]."""
    return _resnet(BasicBlock, [2, 2, 2, 2], num_classes, pretrained, **kwargs)


def resnet34(num_classes: int = 1000, pretrained: bool = False, **kwargs) -> ResNet:
    """ResNet-34: BasicBlock × [3, 4, 6, 3]."""
    return _resnet(BasicBlock, [3, 4, 6, 3], num_classes, pretrained, **kwargs)


def resnet50(num_classes: int = 1000, pretrained: bool = False, **kwargs) -> ResNet:
    """ResNet-50: Bottleneck × [3, 4, 6, 3]."""
    return _resnet(Bottleneck, [3, 4, 6, 3], num_classes, pretrained, **kwargs)


def resnet101(num_classes: int = 1000, pretrained: bool = False, **kwargs) -> ResNet:
    """ResNet-101: Bottleneck × [3, 4, 23, 3]."""
    return _resnet(Bottleneck, [3, 4, 23, 3], num_classes, pretrained, **kwargs)


def resnet152(num_classes: int = 1000, pretrained: bool = False, **kwargs) -> ResNet:
    """ResNet-152: Bottleneck × [3, 8, 36, 3]."""
    return _resnet(Bottleneck, [3, 8, 36, 3], num_classes, pretrained, **kwargs)
