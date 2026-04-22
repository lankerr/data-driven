"""
损失函数
  - B-MSE: 按降水强度分级加权的 MSE (解决长尾分布)
  - SSIM Loss: 保持回波形态结构
  - Combined Loss: B-MSE + SSIM 的加权组合
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BMSELoss(nn.Module):
    """
    Balanced MSE Loss (加权MSE)
    对不同降水强度区间赋予不同权重:
      - 无降水区域 (大面积): 权重低
      - 强降水区域 (小面积但关键): 权重高
    这是气象预测中最重要的改进之一
    """

    def __init__(self, thresholds=None, weights=None):
        """
        thresholds: 归一化后的 VIL 阈值列表 (升序)
        weights: 各区间的权重 (len = len(thresholds) + 1)
        """
        super().__init__()
        if thresholds is None:
            # 默认阈值 (归一化到 0-1 范围, 原始像素值 / 255)
            self.thresholds = [16/255, 74/255, 133/255, 160/255, 181/255]
        else:
            self.thresholds = thresholds

        if weights is None:
            self.weights = [1.0, 2.0, 5.0, 10.0, 30.0, 50.0]
        else:
            self.weights = weights

    def forward(self, pred, target):
        """
        pred, target: (B, T, H, W),  值域 [0, 1]
        """
        mse = (pred - target) ** 2

        # 根据 target 值所在区间, 赋予不同权重
        weight_map = torch.ones_like(target) * self.weights[0]

        for i, thresh in enumerate(self.thresholds):
            mask = target >= thresh
            weight_map[mask] = self.weights[i + 1]

        return (weight_map * mse).mean()


class SSIMLoss(nn.Module):
    """
    SSIM Loss: 1 - SSIM
    保持预测回波的结构完整性 (亮度、对比度、结构)
    逐帧计算 SSIM 后取平均
    """

    def __init__(self, window_size=11, channel=1):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.register_buffer("window", self._create_window(window_size))

    def _create_window(self, window_size):
        """创建高斯窗口"""
        sigma = 1.5
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        g = g / g.sum()
        window = g.unsqueeze(1) * g.unsqueeze(0)  # 2D 高斯
        return window.unsqueeze(0).unsqueeze(0)   # (1, 1, K, K)

    def _ssim(self, img1, img2):
        """计算单通道 SSIM"""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        window = self.window.to(img1.device)

        mu1 = F.conv2d(img1, window, padding=self.window_size // 2)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu12 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 ** 2, window,
                              padding=self.window_size // 2) - mu1_sq
        sigma2_sq = F.conv2d(img2 ** 2, window,
                              padding=self.window_size // 2) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window,
                            padding=self.window_size // 2) - mu12

        ssim = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim.mean()

    def forward(self, pred, target):
        """
        pred, target: (B, T, H, W)
        逐帧计算 SSIM 后平均
        """
        B, T, H, W = pred.shape
        total_ssim = 0.0

        for t in range(T):
            p = pred[:, t:t+1, :, :]
            g = target[:, t:t+1, :, :]
            total_ssim += self._ssim(p, g)

        return 1.0 - total_ssim / T


class CombinedLoss(nn.Module):
    """
    组合损失: L = (1-α) * B-MSE + α * SSIM_Loss
    兼顾像素精度和结构保真
    """

    def __init__(self, thresholds=None, weights=None, ssim_weight=0.3):
        super().__init__()
        self.bmse = BMSELoss(thresholds, weights)
        self.ssim = SSIMLoss()
        self.ssim_weight = ssim_weight

    def forward(self, pred, target):
        l_bmse = self.bmse(pred, target)
        l_ssim = self.ssim(pred, target)
        return (1 - self.ssim_weight) * l_bmse + self.ssim_weight * l_ssim


class MSESSIMLoss(nn.Module):
    """
    纯 MSE + SSIM 组合损失 (与 Earthformer 官方对齐, 不含 B-MSE 分级加权).
    L = (1 - α) * MSE + α * SSIM_Loss

    适用场景:
      - 希望训练信号更 "干净", 不被强降水区域的高权重主导
      - 作为 Mythos-Earthformer 等新架构的默认基线损失
    """

    def __init__(self, ssim_weight: float = 0.3):
        super().__init__()
        self.mse = nn.MSELoss()
        self.ssim = SSIMLoss()
        self.ssim_weight = ssim_weight

    def forward(self, pred, target):
        l_mse = self.mse(pred, target)
        l_ssim = self.ssim(pred, target)
        return (1 - self.ssim_weight) * l_mse + self.ssim_weight * l_ssim


class ResidualLoss(nn.Module):
    """
    残差模型专用损失:
    L = L_final + β * L_base
    同时约束最终预测和基础预测的质量
    """

    def __init__(self, base_criterion=None, final_criterion=None, beta=0.3):
        super().__init__()
        self.base_loss = base_criterion or CombinedLoss()
        self.final_loss = final_criterion or CombinedLoss()
        self.beta = beta

    def forward(self, y_final, y_base, target):
        l_final = self.final_loss(y_final, target)
        l_base = self.base_loss(y_base, target)
        return l_final + self.beta * l_base


def build_loss(cfg):
    """根据配置创建损失函数

    bmse_thresholds 支持两种单位:
      - 整数 (如 [16,74,133,160,181]) → 自动除以 255 归一化
      - 浮点 (如 [0.012,0.018,0.031,0.047,0.135]) 且 max < 1.5 → 视为已归一化
    """
    loss_type = cfg["training"]["loss"]
    raw_thresh = cfg["training"].get("bmse_thresholds", [16, 74, 133, 160, 181])
    if max(raw_thresh) > 1.5:
        thresholds = [t / 255.0 for t in raw_thresh]
    else:
        thresholds = list(raw_thresh)
    weights = list(cfg["training"].get("bmse_weights", [1.0, 2.0, 5.0, 10.0, 30.0]))
    # 权重数量应比阈值多1
    if len(weights) <= len(thresholds):
        weights.append(weights[-1] * 1.5)
    ssim_w = cfg["training"].get("ssim_weight", 0.3)

    if loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "bmse":
        return BMSELoss(thresholds, weights)
    elif loss_type == "ssim":
        return SSIMLoss()
    elif loss_type == "combined":
        return CombinedLoss(thresholds, weights, ssim_w)
    elif loss_type == "mse_ssim":
        return MSESSIMLoss(ssim_weight=ssim_w)
    else:
        raise ValueError(f"未知损失函数: {loss_type}")
