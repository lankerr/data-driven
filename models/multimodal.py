"""
多模态融合模型
利用 SEVIR 的多源数据: VIL(雷达) + C02(可见光) + C09(水汽) + C13(红外) + GLM(闪电)

三种融合策略:
  - 早期融合: 在输入端拼接所有模态 → 单一编码器
  - 中期融合: 独立编码器 → 交叉注意力 → 融合解码
  - 晚期融合: 独立模型 → 加权集成
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .components import ResNetEncoder, SwinEncoder, UNetDecoder


# ============================================================
#  早期融合 (Early Fusion)
# ============================================================
class EarlyFusionModel(nn.Module):
    """
    最简单直接: 将所有模态在通道维度拼接后送入单一编码器
    输入: N个模态 × T帧 → N*T 通道
    优点: 实现简单, 让网络自动学习跨模态关系
    缺点: 所有模态被同等对待, 缺乏针对性
    """

    def __init__(self, modalities=("vil", "vis", "ir069", "ir107"),
                 in_frames=13, out_frames=13,
                 encoder_type="resnet", backbone="resnet34",
                 pretrained=True, **kwargs):
        super().__init__()
        self.modalities = modalities
        total_in_channels = len(modalities) * in_frames  # e.g., 4 * 13 = 52

        if encoder_type == "resnet":
            self.encoder = ResNetEncoder(total_in_channels, backbone, pretrained)
        else:
            self.encoder = SwinEncoder(total_in_channels, backbone, pretrained)

        self.decoder = UNetDecoder(self.encoder.channels, out_frames)

    def forward(self, inputs):
        """inputs: dict {modal_name: (B, T, H, W)} 或 (B, C_total, H, W)"""
        if isinstance(inputs, dict):
            x = torch.cat([inputs[m] for m in self.modalities], dim=1)
        else:
            x = inputs
        features = self.encoder(x)
        return self.decoder(features)


# ============================================================
#  中期融合 (Mid Fusion) -- 交叉注意力
# ============================================================
class CrossModalAttention(nn.Module):
    """交叉模态注意力: 用一个模态的特征去查询另一个模态"""

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, query_feat, kv_feat):
        """
        query_feat, kv_feat: (B, C, H, W)
        将空间维度展平为序列, 做交叉注意力后恢复
        """
        B, C, H, W = query_feat.shape
        q = query_feat.flatten(2).permute(0, 2, 1)  # (B, HW, C)
        kv = kv_feat.flatten(2).permute(0, 2, 1)

        # 交叉注意力
        attn_out, _ = self.attn(q, kv, kv)
        q = self.norm1(q + attn_out)
        q = self.norm2(q + self.ffn(q))

        return q.permute(0, 2, 1).reshape(B, C, H, W)


class MidFusionModel(nn.Module):
    """
    中期融合: 每个模态独立编码 → 在 bottleneck 层做交叉注意力 → 融合解码
    优点: 能捕捉复杂的跨模态互补关系
    缺点: 计算开销大, 需要更多显存
    """

    def __init__(self, modalities=("vil", "vis", "ir069", "ir107"),
                 in_frames=13, out_frames=13,
                 encoder_type="resnet", backbone="resnet34",
                 pretrained=True, num_heads=8, **kwargs):
        super().__init__()
        self.modalities = modalities

        # 每个模态一个独立编码器
        self.encoders = nn.ModuleDict()
        for modal in modalities:
            if encoder_type == "resnet":
                self.encoders[modal] = ResNetEncoder(in_frames, backbone, pretrained)
            else:
                self.encoders[modal] = SwinEncoder(in_frames, backbone, pretrained)

        sample_enc = self.encoders[modalities[0]]
        bottleneck_ch = sample_enc.channels[-1]

        # 交叉注意力融合: 主模态(VIL)与其他模态交互
        self.cross_attns = nn.ModuleList([
            CrossModalAttention(bottleneck_ch, num_heads)
            for _ in range(len(modalities) - 1)
        ])

        # 融合后通道整合
        self.fusion_conv = nn.Conv2d(
            bottleneck_ch * len(modalities), bottleneck_ch, 1
        )

        # 解码器 (使用主模态 VIL 的跳跃连接)
        self.decoder = UNetDecoder(sample_enc.channels, out_frames)

    def forward(self, inputs):
        """inputs: dict {modal_name: (B, T, H, W)}"""
        all_features = {}
        for modal in self.modalities:
            all_features[modal] = self.encoders[modal](inputs[modal])

        # 主模态: VIL (或第一个模态)
        main_modal = self.modalities[0]
        main_bottleneck = all_features[main_modal][-1]

        # 交叉注意力: VIL 查询其他模态
        enhanced = [main_bottleneck]
        other_modals = [m for m in self.modalities if m != main_modal]
        for i, modal in enumerate(other_modals):
            cross_feat = self.cross_attns[i](
                main_bottleneck, all_features[modal][-1]
            )
            enhanced.append(cross_feat)

        # 融合所有模态的 bottleneck 特征
        fused_bottleneck = self.fusion_conv(torch.cat(enhanced, dim=1))

        # 用主模态的跳跃连接 + 融合 bottleneck 解码
        features = all_features[main_modal][:-1] + [fused_bottleneck]
        return self.decoder(features)


# ============================================================
#  晚期融合 (Late Fusion)
# ============================================================
class LateFusionModel(nn.Module):
    """
    晚期融合: 独立模型分别预测 → 学习最优加权组合
    优点: 最稳定, 容易调试, 可逐步添加模态
    缺点: 模型间缺乏交互, 参数量最大
    """

    def __init__(self, modalities=("vil", "vis", "ir069", "ir107"),
                 in_frames=13, out_frames=13,
                 encoder_type="resnet", backbone="resnet34",
                 pretrained=True, **kwargs):
        super().__init__()
        self.modalities = modalities
        n_modals = len(modalities)

        # 每个模态一个完整的 Encoder-Decoder
        self.branches = nn.ModuleDict()
        for modal in modalities:
            if encoder_type == "resnet":
                enc = ResNetEncoder(in_frames, backbone, pretrained)
            else:
                enc = SwinEncoder(in_frames, backbone, pretrained)
            dec = UNetDecoder(enc.channels, out_frames)
            self.branches[modal] = nn.ModuleDict({"enc": enc, "dec": dec})

        # 可学习的融合权重 (1x1 卷积做逐像素加权)
        self.fusion = nn.Sequential(
            nn.Conv2d(out_frames * n_modals, out_frames * 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_frames * 2, out_frames, 1),
        )

    def forward(self, inputs):
        """inputs: dict {modal_name: (B, T, H, W)}"""
        predictions = []
        for modal in self.modalities:
            enc = self.branches[modal]["enc"]
            dec = self.branches[modal]["dec"]
            feat = enc(inputs[modal])
            pred = dec(feat)
            predictions.append(pred)

        # 拼接所有预测, 通过学习的权重融合
        stacked = torch.cat(predictions, dim=1)  # (B, T*N, H, W)
        return self.fusion(stacked)
