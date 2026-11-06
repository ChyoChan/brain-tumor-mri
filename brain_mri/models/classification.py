"""Single-view and dual-input ResNet50 architectures."""

import torch
import torch.nn as nn
from torchvision import models


def build_resnet50(num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_resnet50_dual_input(num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    """ResNet50 that accepts 6 channels: cropped-patch RGB plus original RGB."""
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)
    old_conv = model.conv1
    new_conv = nn.Conv2d(
        6,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )
    with torch.no_grad():
        new_conv.weight[:, :3] = old_conv.weight
        new_conv.weight[:, 3:] = old_conv.weight
        new_conv.weight *= 0.5
    model.conv1 = new_conv
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
