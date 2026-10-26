"""
model_builder.py
================
DeepLabV3+ (Chen et al., ECCV 2018) with an ImageNet-pretrained ResNet-50 encoder for Cataract-1K.

Architecture (Version 1 spec, section 25):
  - Encoder:  torchvision ResNet-50, ImageNet weights (IMAGENET1K_V2), output stride 16
              (layer4 uses dilation 2 instead of stride 2).
  - ASPP:     atrous rates 6 / 12 / 18 + image pooling, 256 channels.
  - Decoder:  ASPP features upsampled x4 and fused with 48-channel projected low-level features
              from layer1 (stride 4), two 3x3 conv blocks, 1x1 classifier, bilinear x4 upsample.
  - Input:    3 x 512 x 512, output 13 logits per pixel.

CataractSegmenter wraps the network with ImageNet normalisation, so the model's input is an RGB
image in [0, 1]. Attacks therefore perturb and clip in pixel space before normalisation.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models.segmentation.deeplabv3 import ASPP

from cataract_dataset import IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES


def _conv_bn_relu(in_ch: int, out_ch: int, k: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, padding=k // 2, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class DeepLabV3PlusResNet50(nn.Module):
    """DeepLabV3+ with a ResNet-50 encoder at output stride 16. Takes normalised input."""

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained_backbone: bool = True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained_backbone else None
        # Output stride 16: keep strides in layer2/layer3, dilate layer4.
        backbone = resnet50(weights=weights, replace_stride_with_dilation=[False, False, True])
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1  # stride 4,  256 ch (low-level features)
        self.layer2 = backbone.layer2  # stride 8,  512 ch
        self.layer3 = backbone.layer3  # stride 16, 1024 ch
        self.layer4 = backbone.layer4  # stride 16, 2048 ch (dilated)

        self.aspp = ASPP(2048, [6, 12, 18], out_channels=256)
        self.low_level_proj = _conv_bn_relu(256, 48, 1)
        self.decoder = nn.Sequential(
            _conv_bn_relu(256 + 48, 256, 3),
            _conv_bn_relu(256, 256, 3),
        )
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

    def encoder_parameters(self):
        for m in (self.stem, self.layer1, self.layer2, self.layer3, self.layer4):
            yield from m.parameters()

    def decoder_parameters(self):
        for m in (self.aspp, self.low_level_proj, self.decoder, self.classifier):
            yield from m.parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        low = self.layer1(self.stem(x))
        high = self.layer4(self.layer3(self.layer2(low)))
        high = self.aspp(high)
        high = F.interpolate(high, size=low.shape[-2:], mode="bilinear", align_corners=False)
        feats = self.decoder(torch.cat([high, self.low_level_proj(low)], dim=1))
        logits = self.classifier(feats)
        return F.interpolate(logits, size=size, mode="bilinear", align_corners=False)


class CataractSegmenter(nn.Module):
    """[0, 1] RGB input -> ImageNet normalisation -> DeepLabV3+ -> logits [B, C, H, W]."""

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained_backbone: bool = True):
        super().__init__()
        self.net = DeepLabV3PlusResNet50(num_classes, pretrained_backbone)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        return self.net((x01 - self.mean) / self.std)


def build_cataract_model(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    device=None,
) -> CataractSegmenter:
    model = CataractSegmenter(num_classes=num_classes, pretrained_backbone=pretrained)
    return model.to(device) if device is not None else model


def load_trained_model(ckpt_path: str, device) -> CataractSegmenter:
    """Load a checkpoint written by train_baseline.py, in eval mode with frozen parameters."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = CataractSegmenter(num_classes=ckpt.get("num_classes", NUM_CLASSES),
                              pretrained_backbone=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_cataract_model(pretrained=True, device=device).eval()
    total, _ = count_parameters(model)
    x = torch.rand(1, 3, 512, 512, device=device)
    with torch.no_grad():
        out = model(x)
    print(f"DeepLabV3+ / ResNet-50 (OS16): {total / 1e6:.2f}M parameters")
    print(f"Input {list(x.shape)} -> output {list(out.shape)}")
    assert out.shape == (1, NUM_CLASSES, 512, 512)
    print("[SUCCESS] forward pass verified")
