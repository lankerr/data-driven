"""
基础组件: 编码器 (ResNet / Swin Transformer) + UNet 解码器
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ============================================================
#  通道适配器: 将任意输入通道数映射到预训练模型所需的3通道
# ============================================================
class ChannelAdapter(nn.Module):
    """1x1 卷积, 将 in_channels 帧映射为 3 通道 (匹配 ImageNet 预训练)"""

    def __init__(self, in_channels, out_channels=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# ============================================================
#  ResNet 编码器
# ============================================================
class ResNetEncoder(nn.Module):
    """
    预训练 ResNet 作为 UNet 编码器, 提取 4 级多尺度特征
    输出特征图:
      f1: (B, 64,  H/4,  W/4)   -- ResNet34
      f2: (B, 128, H/8,  W/8)
      f3: (B, 256, H/16, W/16)
      f4: (B, 512, H/32, W/32)  -- bottleneck
    """

    def __init__(self, in_channels=13, backbone="resnet34", pretrained=True, reduce_stem=False):
        super().__init__()
        self.adapter = ChannelAdapter(in_channels, 3)

        weights = "IMAGENET1K_V1" if pretrained else None
        if backbone == "resnet34":
            resnet = models.resnet34(weights=weights)
            self.channels = [64, 128, 256, 512]
        elif backbone == "resnet50":
            resnet = models.resnet50(weights=weights)
            self.channels = [256, 512, 1024, 2048]
        else:
            raise ValueError(f"不支持的 backbone: {backbone}")

        # 拆解 ResNet 为各阶段
        if reduce_stem:
            # 去掉 maxpool, stem 只下采样 /2 (保留更多空间分辨率)
            self.stem = nn.Sequential(
                resnet.conv1, resnet.bn1, resnet.relu
            )  # /2
        else:
            self.stem = nn.Sequential(
                resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
            )  # /4
        self.layer1 = resnet.layer1  # /4
        self.layer2 = resnet.layer2  # /8
        self.layer3 = resnet.layer3  # /16
        self.layer4 = resnet.layer4  # /32

    def forward(self, x):
        x = self.adapter(x)
        x = self.stem(x)
        f1 = self.layer1(x)   # (B, 64,  H/4,  W/4)
        f2 = self.layer2(f1)  # (B, 128, H/8,  W/8)
        f3 = self.layer3(f2)  # (B, 256, H/16, W/16)
        f4 = self.layer4(f3)  # (B, 512, H/32, W/32)
        return [f1, f2, f3, f4]


# ============================================================
#  Swin Transformer 编码器 (作为 "ViT" 的实际实现)
# ============================================================
class SwinEncoder(nn.Module):
    """
    预训练 Swin Transformer 作为 UNet 编码器
    天然提供 4 级层次化特征, 完美适配 UNet:
      f1: (B, 96,  H/4,  W/4)
      f2: (B, 192, H/8,  W/8)
      f3: (B, 384, H/16, W/16)
      f4: (B, 768, H/32, W/32)  -- bottleneck
    """

    def __init__(self, in_channels=13, backbone="swin_t", pretrained=True, reduce_stem=False):
        super().__init__()
        self.adapter = ChannelAdapter(in_channels, 3)

        weights = "IMAGENET1K_V1" if pretrained else None
        if backbone == "swin_t":
            swin = models.swin_t(weights=weights)
            self.channels = [96, 192, 384, 768]
        elif backbone == "swin_s":
            swin = models.swin_s(weights=weights)
            self.channels = [96, 192, 384, 768]
        else:
            raise ValueError(f"不支持的 backbone: {backbone}")

        # Swin features: [patch_embed, blocks1, merge2, blocks2, merge3, blocks3, merge4, blocks4]
        self.stage1 = nn.Sequential(swin.features[0], swin.features[1])
        self.stage2 = nn.Sequential(swin.features[2], swin.features[3])
        self.stage3 = nn.Sequential(swin.features[4], swin.features[5])
        self.stage4 = nn.Sequential(swin.features[6], swin.features[7])

        if reduce_stem:
            # 替换 patch embedding: stride 4→2, 保留更多空间分辨率 (/2 而非 /4)
            embed_dim = self.channels[0]
            self.stage1[0][0] = nn.Conv2d(3, embed_dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.adapter(x)
        # Swin 内部格式: (B, H, W, C) -- torchvision 实现
        f1 = self.stage1(x)   # (B, H/4,  W/4,  96)
        f2 = self.stage2(f1)  # (B, H/8,  W/8,  192)
        f3 = self.stage3(f2)  # (B, H/16, W/16, 384)
        f4 = self.stage4(f3)  # (B, H/32, W/32, 768)

        # 转为标准 (B, C, H, W) 格式
        f1 = f1.permute(0, 3, 1, 2).contiguous()
        f2 = f2.permute(0, 3, 1, 2).contiguous()
        f3 = f3.permute(0, 3, 1, 2).contiguous()
        f4 = f4.permute(0, 3, 1, 2).contiguous()

        return [f1, f2, f3, f4]


# ============================================================
#  UNet 解码器模块
# ============================================================
class UpBlock(nn.Module):
    """UNet 上采样模块: 上采样 + 拼接跳跃连接 + 双卷积"""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2,
                                      kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch // 2 + skip_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        # 处理尺寸不一致
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear",
                              align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetDecoder(nn.Module):
    """
    通用 UNet 解码器, 适配不同编码器的通道配置
    encoder_channels: 编码器各级通道数 [浅→深], 如 [64, 128, 256, 512]
    """

    def __init__(self, encoder_channels, out_channels=13, decoder_channels=None, stem_stride=4):
        super().__init__()

        if decoder_channels is None:
            decoder_channels = [ch // 2 for ch in reversed(encoder_channels[:-1])]
            # ResNet34: [128, 64, 32]

        enc_rev = list(reversed(encoder_channels))  # 深→浅: [512, 256, 128, 64]
        bottleneck_ch = enc_rev[0]
        skip_channels = enc_rev[1:]  # [256, 128, 64]

        self.up_blocks = nn.ModuleList()
        ch = bottleneck_ch
        for skip_ch, dec_ch in zip(skip_channels, decoder_channels):
            self.up_blocks.append(UpBlock(ch, skip_ch, dec_ch))
            ch = dec_ch

        # 最终上采样: 从 H/stem_stride 恢复到 H
        if stem_stride == 2:
            # reduce_stem 模式: 只需 ×2 上采样
            self.final = nn.Sequential(
                nn.ConvTranspose2d(ch, ch // 2, kernel_size=2, stride=2),
                nn.BatchNorm2d(ch // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch // 2, out_channels, kernel_size=1),
            )
        else:
            # 默认: 需要 ×4 上采样
            self.final = nn.Sequential(
                nn.ConvTranspose2d(ch, ch // 2, kernel_size=2, stride=2),
                nn.BatchNorm2d(ch // 2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(ch // 2, ch // 4, kernel_size=2, stride=2),
                nn.BatchNorm2d(ch // 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch // 4, out_channels, kernel_size=1),
            )

    def forward(self, features):
        """
        features: [f1, f2, f3, f4] 从浅到深
        f4 是 bottleneck, f1-f3 是跳跃连接
        """
        skips = features[:-1]          # [f1, f2, f3]
        skips_rev = list(reversed(skips))  # [f3, f2, f1]
        x = features[-1]              # f4 (bottleneck)

        for up_block, skip in zip(self.up_blocks, skips_rev):
            x = up_block(x, skip)

        x = self.final(x)
        return x


# ============================================================
#  工具函数
# ============================================================
def freeze_encoder(encoder):
    """冻结编码器参数 (只训练 adapter 和 decoder)"""
    for name, param in encoder.named_parameters():
        if "adapter" not in name:
            param.requires_grad = False


def count_parameters(model):
    """统计可训练参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
