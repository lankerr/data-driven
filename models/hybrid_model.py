"""
实验 5: 混合融合架构 (Hybrid UNet)
在同一个 UNet 中有机融合 ResNet 和 Swin Transformer

策略:
  - 双分支编码器: ResNet 和 Swin 并行提取特征
  - 特征融合门控: 学习每个尺度上两种特征的最优混合比例
  - 共享解码器: 融合后的特征通过统一解码器重建
"""
import torch
import torch.nn as nn
from .components import ResNetEncoder, SwinEncoder, UNetDecoder


class FeatureFusionGate(nn.Module):
    """
    门控融合模块: 学习 ResNet 和 Swin 特征的最优混合权重
    gate = σ(W_r·f_resnet + W_s·f_swin + b)
    f_fused = gate * f_resnet + (1 - gate) * f_swin
    """

    def __init__(self, ch_resnet, ch_swin, out_ch):
        super().__init__()
        # 先将两种特征统一到相同通道数
        self.proj_resnet = nn.Conv2d(ch_resnet, out_ch, 1)
        self.proj_swin = nn.Conv2d(ch_swin, out_ch, 1)
        # 门控权重
        self.gate = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, f_resnet, f_swin):
        r = self.proj_resnet(f_resnet)
        s = self.proj_swin(f_swin)

        # 确保空间尺寸一致
        if r.shape[2:] != s.shape[2:]:
            s = nn.functional.interpolate(s, size=r.shape[2:],
                                          mode="bilinear", align_corners=False)

        g = self.gate(torch.cat([r, s], dim=1))  # (B, out_ch, H, W)
        fused = g * r + (1 - g) * s
        return fused


class HybridUNet(nn.Module):
    """
    实验5: ResNet + Swin Transformer 有机融合
    两个编码器并行运行, 在每个尺度级别通过门控机制融合特征
    """

    def __init__(self, in_channels=13, out_channels=13,
                 resnet_backbone="resnet34", vit_backbone="swin_t",
                 pretrained=True, reduce_stem=False, **kwargs):
        super().__init__()
        stem_stride = 2 if reduce_stem else 4

        self.resnet_enc = ResNetEncoder(in_channels, resnet_backbone, pretrained, reduce_stem=reduce_stem)
        self.swin_enc = SwinEncoder(in_channels, vit_backbone, pretrained, reduce_stem=reduce_stem)

        r_ch = self.resnet_enc.channels  # [64, 128, 256, 512]
        s_ch = self.swin_enc.channels    # [96, 192, 384, 768]

        # 每个尺度的融合通道数: 取较大者
        fused_ch = [max(r, s) for r, s in zip(r_ch, s_ch)]
        # 如 [96, 192, 384, 768]

        # 4 级门控融合
        self.fusion_gates = nn.ModuleList([
            FeatureFusionGate(rc, sc, fc)
            for rc, sc, fc in zip(r_ch, s_ch, fused_ch)
        ])

        # 解码器使用融合后的通道数
        self.decoder = UNetDecoder(fused_ch, out_channels, stem_stride=stem_stride)

    def forward(self, x):
        # 双分支编码
        r_features = self.resnet_enc(x)   # [f1_r, f2_r, f3_r, f4_r]
        s_features = self.swin_enc(x)     # [f1_s, f2_s, f3_s, f4_s]

        # 逐级门控融合
        fused = []
        for gate, fr, fs in zip(self.fusion_gates, r_features, s_features):
            fused.append(gate(fr, fs))

        # 解码
        return self.decoder(fused)
