"""
实验 3 & 4: 残差预测架构
  - 实验3: ResNet 打底 + ViT 预测残差
  - 实验4: ViT 打底 + ResNet 预测残差 (⭐ 最推荐)

核心思想:
  第一个网络预测基础预测 Y_base
  第二个网络接收 [原始输入, Y_base] 预测残差 ΔY
  最终输出 Y_final = Y_base + ΔY
"""
import torch
import torch.nn as nn
from .components import ResNetEncoder, SwinEncoder, UNetDecoder, freeze_encoder


class _ResidualModel(nn.Module):
    """残差预测的通用基类"""

    def __init__(self, base_encoder, refine_encoder,
                 in_channels, out_channels):
        super().__init__()
        # 基础网络: 输入原始帧
        self.base_encoder = base_encoder
        self.base_decoder = UNetDecoder(base_encoder.channels, out_channels)

        # 残差网络: 输入 [原始帧, 基础预测] = 2 * channels
        self.refine_encoder = refine_encoder
        self.refine_decoder = UNetDecoder(refine_encoder.channels, out_channels)

    def forward(self, x):
        # Stage 1: 基础预测
        base_features = self.base_encoder(x)
        y_base = self.base_decoder(base_features)

        # Stage 2: 残差预测
        # 将原始输入和基础预测拼接作为残差网络的输入
        refine_input = torch.cat([x, y_base.detach()], dim=1)
        refine_features = self.refine_encoder(refine_input)
        delta_y = self.refine_decoder(refine_features)

        # Y_final = Y_base + ΔY
        y_final = y_base + delta_y

        return y_final, y_base  # 同时返回基础预测用于辅助损失


class ResNetViTResidual(nn.Module):
    """
    实验3: ResNet 预测基础 + ViT 预测残差
    直觉: CNN 先画出粗略的回波轮廓, Transformer 补充全局修正
    """

    def __init__(self, in_channels=13, out_channels=13,
                 resnet_backbone="resnet34", vit_backbone="swin_t",
                 pretrained=True, freeze_base=False, reduce_stem=False, **kwargs):
        super().__init__()
        stem_stride = 2 if reduce_stem else 4

        self.base_encoder = ResNetEncoder(in_channels, resnet_backbone, pretrained, reduce_stem=reduce_stem)
        self.base_decoder = UNetDecoder(self.base_encoder.channels, out_channels, stem_stride=stem_stride)

        # 残差网络输入: 原始帧 + 基础预测 = in_channels + out_channels
        self.refine_encoder = SwinEncoder(in_channels + out_channels,
                                           vit_backbone, pretrained, reduce_stem=reduce_stem)
        self.refine_decoder = UNetDecoder(self.refine_encoder.channels, out_channels, stem_stride=stem_stride)

        if freeze_base:
            freeze_encoder(self.base_encoder)

    def forward(self, x):
        # Stage 1: ResNet 基础预测
        base_feat = self.base_encoder(x)
        y_base = self.base_decoder(base_feat)

        # Stage 2: ViT 残差预测
        refine_in = torch.cat([x, y_base], dim=1)
        refine_feat = self.refine_encoder(refine_in)
        delta_y = self.refine_decoder(refine_feat)

        y_final = y_base + delta_y
        return y_final, y_base


class ViTResNetResidual(nn.Module):
    """
    实验4: ViT 预测基础 + ResNet 预测残差 ⭐
    直觉: Transformer 先把回波的大尺度移动和全局位置预测对,
          ResNet 再去修补局部对流细节 (突发暴雨核心等高频信息)
    这在气象物理上最自洽: 先有大气环流驱动, 再有局部对流爆发
    """

    def __init__(self, in_channels=13, out_channels=13,
                 resnet_backbone="resnet34", vit_backbone="swin_t",
                 pretrained=True, freeze_base=False, reduce_stem=False, **kwargs):
        super().__init__()
        stem_stride = 2 if reduce_stem else 4

        self.base_encoder = SwinEncoder(in_channels, vit_backbone, pretrained, reduce_stem=reduce_stem)
        self.base_decoder = UNetDecoder(self.base_encoder.channels, out_channels, stem_stride=stem_stride)

        # 残差网络输入: 原始帧 + 基础预测
        self.refine_encoder = ResNetEncoder(in_channels + out_channels,
                                             resnet_backbone, pretrained, reduce_stem=reduce_stem)
        self.refine_decoder = UNetDecoder(self.refine_encoder.channels, out_channels, stem_stride=stem_stride)

        if freeze_base:
            freeze_encoder(self.base_encoder)

    def forward(self, x):
        # Stage 1: ViT 全局预测
        base_feat = self.base_encoder(x)
        y_base = self.base_decoder(base_feat)

        # Stage 2: ResNet 局部残差修补
        refine_in = torch.cat([x, y_base], dim=1)
        refine_feat = self.refine_encoder(refine_in)
        delta_y = self.refine_decoder(refine_feat)

        y_final = y_base + delta_y
        return y_final, y_base
