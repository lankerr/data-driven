"""
实验 1 & 2: 单骨干网络 UNet
  - ResNet-UNet: 预训练 ResNet 编码器 + UNet 解码器
  - ViT-UNet:    预训练 Swin Transformer 编码器 + UNet 解码器
"""
import torch.nn as nn
from .components import ResNetEncoder, SwinEncoder, UNetDecoder, freeze_encoder


class ResNetUNet(nn.Module):
    """
    实验1: 纯 ResNet-UNet
    直觉: ResNet 擅长捕捉局部高分辨率细节 (对流核心、回波边缘)
    预期: 边缘清晰, 但长距离平流预测能力有限
    """

    def __init__(self, in_channels=13, out_channels=13,
                 backbone="resnet34", pretrained=True,
                 freeze=False, reduce_stem=False, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, backbone, pretrained, reduce_stem=reduce_stem)
        stem_stride = 2 if reduce_stem else 4
        self.decoder = UNetDecoder(self.encoder.channels, out_channels, stem_stride=stem_stride)

        if freeze:
            freeze_encoder(self.encoder)

    def forward(self, x):
        features = self.encoder(x)
        return self.decoder(features)


class ViTUNet(nn.Module):
    """
    实验2: 纯 ViT-UNet (使用 Swin Transformer)
    直觉: Transformer 擅长全局依赖建模 (大尺度平流、环流场)
    预期: 整体位置准确, 但局部细节可能模糊
    """

    def __init__(self, in_channels=13, out_channels=13,
                 backbone="swin_t", pretrained=True,
                 freeze=False, reduce_stem=False, **kwargs):
        super().__init__()
        self.encoder = SwinEncoder(in_channels, backbone, pretrained, reduce_stem=reduce_stem)
        stem_stride = 2 if reduce_stem else 4
        self.decoder = UNetDecoder(self.encoder.channels, out_channels, stem_stride=stem_stride)

        if freeze:
            freeze_encoder(self.encoder)

    def forward(self, x):
        features = self.encoder(x)
        return self.decoder(features)
