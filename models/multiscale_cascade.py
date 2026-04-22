"""
多尺度时间金字塔 (Multi-Scale Temporal Pyramid)
===============================================
四级时间金字塔, 全部输入 <= 30 帧, 计算量与单个 ViT-UNet 同量级.

past 侧从 `in_frames=210` (21h @ 6min) 的历史窗口按 4 个 stride 采样 (end-aligned):
    L0 (stride 30 / 3h):    past[29::30]         = 7 帧  → 预测 1 帧   (未来3h终点)
    L1 (stride 10 / 1h):    past[9::10]          = 21 帧 → 预测 3 帧   (1h/2h/3h)
    L2 (stride  5 / 30min): past[4::5][-30:]     = 30 帧 → 预测 6 帧   (30m .. 3h)
    L3 (stride  1 / 6min):  past[-30:]           = 30 帧 → 预测 30 帧  (6m 全分辨率)

future (B, 30, H, W) 在 4 个尺度上切片作深度监督:
    L0: future[29:30]      (1帧)
    L1: future[9::10]      (3帧)
    L2: future[4::5]       (6帧)
    L3: future             (30帧)

级间连接: UNet-skip + residual
    y_prev → 时间插值到本级帧数 → [可选 detach] → concat 到输入 → ViT-UNet → delta
    y_cur = y_prev_up + delta

detach 开关:
    False → 梯度跨级贯通 (端到端深度监督)
    True  → 每级独立梯度, 但数值 (concat + 残差锚) 仍前向贯通
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .single_model import ViTUNet


# 四个尺度: (stride, n_past, n_future)
# past 从 (B, 210, H, W) 中按 stride end-aligned 切, 最多取 n_past 帧 (取最近的)
SCALE_SPECS = [
    (30, 7,  1),     # L0: 3h    (7 * 3h = 21h past)
    (10, 21, 3),     # L1: 1h    (21 * 1h = 21h past)
    (5,  30, 6),     # L2: 30min (30 * 0.5h = 15h past)
    (1,  30, 30),    # L3: 6min  (30 * 0.1h = 3h past)
]
PAST_FRAMES = 210    # 21h


def _time_interp(x, T_out):
    """(B, T, H, W) -> (B, T_out, H, W) 时间维线性插值."""
    B, T, H, W = x.shape
    if T == T_out:
        return x
    if T == 1:
        return x.expand(B, T_out, H, W).contiguous()
    x_t = x.permute(0, 2, 3, 1).reshape(B * H * W, 1, T)
    x_t = F.interpolate(x_t, size=T_out, mode="linear", align_corners=True)
    return x_t.reshape(B, H, W, T_out).permute(0, 3, 1, 2).contiguous()


def _slice_past(past, stride, n_past):
    """end-aligned 切片: 从 past 末尾向前取 n_past 帧 @ stride."""
    # 所有可能的 end-aligned 索引 (把 past 末尾的 frame 当 t=-1)
    # 取最后 n_past 个 stride-间隔点
    T = past.shape[1]
    # 索引: [T-1, T-1-stride, T-1-2*stride, ...] 共 n_past 个, 升序
    idx = [T - 1 - i * stride for i in range(n_past)][::-1]
    assert idx[0] >= 0, f"past 太短: T={T}, stride={stride}, n_past={n_past}"
    return past[:, idx]


def _slice_future(future, stride, n_future):
    """end-aligned 切片: future[stride-1::stride]."""
    assert future.shape[1] == n_future * stride, \
        f"future {future.shape[1]} != {n_future}*{stride}"
    return future[:, stride - 1::stride]


class MultiScaleTemporalCascade(nn.Module):
    """
    4 级时间金字塔, 所有子模型都是 ViT-UNet (Swin-T + UNet).
    """

    def __init__(self, use_fine=True,
                 detach_between_stages=False,
                 vit_backbone="swin_t", pretrained=True, reduce_stem=True,
                 **kwargs):
        super().__init__()
        self.use_fine = use_fine
        self.detach = detach_between_stages

        common = dict(backbone=vit_backbone, pretrained=pretrained,
                      reduce_stem=reduce_stem)

        # L0: past 7 → future 1
        _, p0, f0 = SCALE_SPECS[0]
        self.L0 = ViTUNet(in_channels=p0, out_channels=f0, **common)

        # L1: past 21 + y0_up(3) → future 3
        _, p1, f1 = SCALE_SPECS[1]
        self.L1 = ViTUNet(in_channels=p1 + f1, out_channels=f1, **common)

        # L2: past 30 + y1_up(6) → future 6
        _, p2, f2 = SCALE_SPECS[2]
        self.L2 = ViTUNet(in_channels=p2 + f2, out_channels=f2, **common)

        # L3: past 30 + y2_up(30) → future 30
        if use_fine:
            _, p3, f3 = SCALE_SPECS[3]
            self.L3 = ViTUNet(in_channels=p3 + f3, out_channels=f3, **common)

    # ------------------------------------------------------------------
    def _cond(self, y_prev, n_out):
        y_up = _time_interp(y_prev, n_out)
        return y_up.detach() if self.detach else y_up

    def forward(self, past):
        """past: (B, 210, H, W) — 过去 21h @ 6min"""
        assert past.shape[1] == PAST_FRAMES, \
            f"expect past with {PAST_FRAMES} frames, got {past.shape[1]}"

        # ----- L0: 3h 格点, 7→1 -----
        s0, p0, f0 = SCALE_SPECS[0]
        x0 = _slice_past(past, s0, p0)           # (B, 7, H, W)
        y0 = self.L0(x0)                         # (B, 1, H, W)

        # ----- L1: 1h 格点, 21+3→3 -----
        s1, p1, f1 = SCALE_SPECS[1]
        x1 = _slice_past(past, s1, p1)           # (B, 21, H, W)
        c1 = self._cond(y0, f1)                  # (B, 3, H, W)
        delta1 = self.L1(torch.cat([x1, c1], dim=1))
        y1 = c1 + delta1

        # ----- L2: 30min 格点, 30+6→6 -----
        s2, p2, f2 = SCALE_SPECS[2]
        x2 = _slice_past(past, s2, p2)           # (B, 30, H, W)
        c2 = self._cond(y1, f2)                  # (B, 6, H, W)
        delta2 = self.L2(torch.cat([x2, c2], dim=1))
        y2 = c2 + delta2

        if not self.use_fine:
            return {"y0": y0, "y1": y1, "y2": y2, "y3": None,
                    "final": _time_interp(y2, 30)}

        # ----- L3: 6min 全分辨率, 30+30→30 -----
        s3, p3, f3 = SCALE_SPECS[3]
        x3 = _slice_past(past, s3, p3)           # (B, 30, H, W) = past 最后 30 帧
        c3 = self._cond(y2, f3)                  # (B, 30, H, W)
        delta3 = self.L3(torch.cat([x3, c3], dim=1))
        y3 = c3 + delta3

        return {"y0": y0, "y1": y1, "y2": y2, "y3": y3, "final": y3}


class MSTCLoss(nn.Module):
    """深度监督: future (B,30,H,W) 在 4 个尺度上切片逐级监督."""

    def __init__(self, base_loss, w_l0=0.1, w_l1=0.2, w_l2=0.3, w_l3=1.0):
        super().__init__()
        self.base = base_loss
        self.w = (w_l0, w_l1, w_l2, w_l3)

    def forward(self, out, future):
        t0 = _slice_future(future, 30, 1)        # (B, 1,  H, W)
        t1 = _slice_future(future, 10, 3)        # (B, 3,  H, W)
        t2 = _slice_future(future, 5,  6)        # (B, 6,  H, W)
        t3 = future                               # (B, 30, H, W)

        loss = self.w[0] * self.base(out["y0"], t0)
        loss = loss + self.w[1] * self.base(out["y1"], t1)
        loss = loss + self.w[2] * self.base(out["y2"], t2)
        if out.get("y3") is not None and self.w[3] > 0:
            loss = loss + self.w[3] * self.base(out["y3"], t3)
        return loss
