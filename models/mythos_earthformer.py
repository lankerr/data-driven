"""
Mythos-Earthformer
==================
基于 Amazon Earthformer (Cuboid Attention + U-Net) 骨架，将 **最底层瓶颈 (Bottleneck)**
改造为 **权重共享的循环推理引擎 (Continuous Latent Space Reasoning / Recurrent Depth)**。

核心思想:
  - 不在每一层都使用独立权重
  - 让高维时空潜变量 Z 在 **同一个 Cuboid Attention Block** 中进行 N 次循环迭代
  - 训练期 N 从区间随机采样 (Random Depth Training), 推理期可固定为大 N 以"深度思考"

宏观数据流 (以 SEVIR 13 in / 12 out / 384x384 为例)::

    [B, 13, 384, 384]               # 原始输入 (dataset 给出)
        └── unsqueeze(C=1) ──►  [B, 13, 384, 384, 1]          # NTHWC 布局
        └── PatchEmbed ──►      [B, 13, 384, 384, 128]
        └── Encoder Scale-1 ──► [B, 13, 384, 384, 128]
        └── TemporalProj 13→12 ──► [B, 12, 384, 384, 128]     # 缓存 Skip_1
        └── SpatialDownsample ──► [B, 12, 192, 192, 256]      # Z_0
        └── RecurrentBottleneck × N (共享权重!) ──► [B, 12, 192, 192, 256]  # Z_N
        └── SpatialUpsample ──► [B, 12, 384, 384, 128]
        └── + Skip_1 (skip connection, add)
        └── Decoder Scale-1 ──► [B, 12, 384, 384, 128]
        └── OutputHead (1x1 conv) ──► [B, 12, 384, 384, 1]
        └── squeeze(C) ──►       [B, 12, 384, 384]

注意力算子支持两种:
  * ``attn_type="cuboid"``  —— 从 earth-forecasting-transformer 仓库加载
                             ``StackCuboidSelfAttentionBlock`` (内置 Shift 机制)
  * ``attn_type="conv3d"``  —— 轻量 Conv3d 替代, 用于 smoke test / 梯度回传验证

训练/推理 N 步的切换逻辑在 ``forward`` 中自动处理:
  * ``self.training == True``  → N 从 ``num_steps_range`` 随机采样
  * ``self.training == False`` → N = ``self.eval_num_steps`` (构造时设定, 可外部覆盖)
"""
from __future__ import annotations

import os
import sys
import random
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp


# ---------------------------------------------------------------------------
# 尝试加载官方 Earthformer 的 Cuboid 算子 (可选)
# ---------------------------------------------------------------------------
_EARTHFORMER_SRC = r"C:\Users\97290\Desktop\MOE\earth-forecasting-transformer\src"
_EARTHFORMER_AVAILABLE = False
try:
    if os.path.isdir(_EARTHFORMER_SRC) and _EARTHFORMER_SRC not in sys.path:
        sys.path.append(_EARTHFORMER_SRC)
    from earthformer.cuboid_transformer.cuboid_transformer import (  # type: ignore
        StackCuboidSelfAttentionBlock,
        PatchMerging3D,
        Upsample3DLayer,
    )
    _EARTHFORMER_AVAILABLE = True
except Exception as _e:  # noqa: BLE001
    _IMPORT_ERR = _e


# ---------------------------------------------------------------------------
# 轻量 3D 卷积替代 (smoke test & 无 earthformer 环境下的降级路径)
# ---------------------------------------------------------------------------
class Conv3DAttnBlock(nn.Module):
    """Cuboid Attention 的轻量占位符。保持 NTHWC 形状不变。"""

    def __init__(self, dim: int, kernel_size: int = 3, dropout: float = 0.0):
        super().__init__()
        pad = kernel_size // 2
        self.norm1 = nn.LayerNorm(dim)
        self.conv1 = nn.Conv3d(dim, dim, kernel_size=kernel_size, padding=pad)
        self.conv2 = nn.Conv3d(dim, dim, kernel_size=kernel_size, padding=pad)
        self.act = nn.GELU()
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B,T,H,W,C]
        # Conv3d 分支 (pre-norm)
        h = self.norm1(x)
        h = h.permute(0, 4, 1, 2, 3).contiguous()  # B,C,T,H,W
        h = self.act(self.conv1(h))
        h = self.conv2(h)
        h = h.permute(0, 2, 3, 4, 1).contiguous()  # B,T,H,W,C
        x = x + h
        # FFN
        x = x + self.ffn(self.norm2(x))
        return x


def _build_attn_block(
    attn_type: str,
    dim: int,
    num_heads: int,
    cuboid_size: Tuple[Tuple[int, int, int], Tuple[int, int, int]],
    shift_size: Tuple[Tuple[int, int, int], Tuple[int, int, int]],
    strategy: Tuple[Tuple[str, str, str], Tuple[str, str, str]],
    ffn_drop: float,
    attn_drop: float,
    use_inter_ffn: bool = True,
) -> nn.Module:
    if attn_type == "cuboid":
        if not _EARTHFORMER_AVAILABLE:
            raise ImportError(
                "attn_type='cuboid' 需要 earth-forecasting-transformer 可被导入; "
                f"当前路径 {_EARTHFORMER_SRC} 无法加载: {_IMPORT_ERR}"  # type: ignore[name-defined]
            )
        return StackCuboidSelfAttentionBlock(
            dim=dim,
            num_heads=num_heads,
            block_cuboid_size=list(cuboid_size),
            block_shift_size=list(shift_size),
            block_strategy=list(strategy),
            padding_type="ignore",
            attn_drop=attn_drop,
            proj_drop=ffn_drop,
            ffn_drop=ffn_drop,
            use_inter_ffn=use_inter_ffn,
            use_relative_pos=True,
        )
    elif attn_type == "conv3d":
        return Conv3DAttnBlock(dim=dim, dropout=ffn_drop)
    else:
        raise ValueError(f"未知 attn_type: {attn_type}")


# ---------------------------------------------------------------------------
# 下/上采样 (无 earthformer 时的 fallback 实现)
# ---------------------------------------------------------------------------
class _FallbackSpatialDown(nn.Module):
    """空间 2x 下采样 + 通道翻倍, NTHWC → NTHWC"""

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Conv3d(dim_in, dim_out,
                              kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.norm = nn.LayerNorm(dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NTHWC → NCTHW
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = self.proj(x)
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        return self.norm(x)


class _FallbackSpatialUp(nn.Module):
    """空间 2x 上采样 + 通道减半, NTHWC → NTHWC"""

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Conv3d(dim_in, dim_out, kernel_size=(1, 3, 3),
                              padding=(0, 1, 1))
        self.norm = nn.LayerNorm(dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H, W, C = x.shape
        # 2D 最近邻上采样 (逐帧)
        x = x.reshape(B * T, H, W, C).permute(0, 3, 1, 2)  # BT,C,H,W
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = x.permute(0, 2, 3, 1).reshape(B, T, H * 2, W * 2, C)
        # 3x3 卷积调整通道
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = self.proj(x)
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        return self.norm(x)


# ---------------------------------------------------------------------------
# 主模型
# ---------------------------------------------------------------------------
class MythosEarthformer(nn.Module):
    """
    Mythos-Earthformer: Earthformer 骨架 + Recurrent Latent Bottleneck。

    Parameters
    ----------
    in_frames : int
        输入帧数 T_in (SEVIR 默认 13)
    out_frames : int
        输出帧数 T_out (SEVIR 默认 12)
    base_dim : int
        基础通道数 (Scale-1 宽度), 默认 128
    num_heads : int
        注意力头数
    num_steps_range : (int, int)
        训练期 N 的随机区间 [low, high] (含两端)
    eval_num_steps : int
        推理期默认 N
    attn_type : {"cuboid", "conv3d"}
        注意力算子类型。"cuboid" 使用官方 Earthformer 算子, "conv3d" 为轻量替代。
    use_checkpoint : bool
        是否对循环单元启用 activation checkpointing (节省显存, 但会重算一次前向)
    in_channels : int
        输入通道数 (SEVIR VIL 为 1)。当前 dataset 返回 [B,T,H,W] 不含 C 维,
        模型内部会自动 unsqueeze/squeeze。如果传入的已是 [B,T,H,W,C], 需要 C == in_channels。
    """

    def __init__(
        self,
        in_frames: int = 13,
        out_frames: int = 12,
        base_dim: int = 128,
        num_heads: int = 4,
        num_steps_range: Tuple[int, int] = (3, 8),
        eval_num_steps: int = 6,
        attn_type: str = "conv3d",
        use_checkpoint: bool = False,
        in_channels: int = 1,
        ffn_drop: float = 0.1,
        attn_drop: float = 0.0,
        # 以下参数仅对 attn_type="cuboid" 有效
        cuboid_size: Tuple[Tuple[int, int, int], Tuple[int, int, int]] =
            ((4, 4, 4), (4, 4, 4)),
        shift_size: Tuple[Tuple[int, int, int], Tuple[int, int, int]] =
            ((0, 0, 0), (2, 2, 2)),
        strategy: Tuple[Tuple[str, str, str], Tuple[str, str, str]] =
            (("l", "l", "l"), ("l", "l", "l")),
        **_unused,
    ):
        super().__init__()
        assert len(num_steps_range) == 2 and num_steps_range[0] >= 1
        assert attn_type in ("cuboid", "conv3d")

        self.in_frames = in_frames
        self.out_frames = out_frames
        self.base_dim = base_dim
        self.in_channels = in_channels
        self.num_steps_range = tuple(num_steps_range)
        self.eval_num_steps = eval_num_steps
        self.attn_type = attn_type
        self.use_checkpoint = use_checkpoint

        d1 = base_dim          # Scale-1 通道数
        d2 = base_dim * 2      # Scale-2 (bottleneck) 通道数

        # ------- 1. Patch Embedding -------
        # 输入 [B,T,H,W,C_in] → [B,T,H,W,d1]
        self.patch_embed = nn.Linear(in_channels, d1)

        # ------- 2. Encoder Scale-1 -------
        self.encoder_block = _build_attn_block(
            attn_type=attn_type, dim=d1, num_heads=num_heads,
            cuboid_size=cuboid_size, shift_size=shift_size, strategy=strategy,
            ffn_drop=ffn_drop, attn_drop=attn_drop,
        )

        # ------- 3. 时间维投影: T_in → T_out (作用在 Skip_1 这一侧) -------
        # 在 T 维度做 Linear: [B,T_in,...] → [B,T_out,...]
        self.temporal_proj = nn.Linear(in_frames, out_frames, bias=False)

        # ------- 4. 空间下采样 Scale-1 → Scale-2 -------
        if _EARTHFORMER_AVAILABLE:
            self.downsample = PatchMerging3D(
                dim=d1, out_dim=d2, downsample=(1, 2, 2), padding_type="nearest",
            )
        else:
            self.downsample = _FallbackSpatialDown(d1, d2)

        # ------- 5. 循环瓶颈 (核心!) -------
        # 只有 **一个** Cuboid Attention Block, N 次调用共享权重
        self.recurrent_cell = _build_attn_block(
            attn_type=attn_type, dim=d2, num_heads=num_heads,
            cuboid_size=cuboid_size, shift_size=shift_size, strategy=strategy,
            ffn_drop=ffn_drop, attn_drop=attn_drop,
        )

        # ------- 6. 空间上采样 Scale-2 → Scale-1 -------
        # target_size 取决于输入 H, W; 用 fallback (基于 interpolate, 尺寸自适应)
        # 注意: earthformer 的 Upsample3DLayer 需要在构造时传入 target_size,
        # 而我们希望 H/W 动态, 因此这里统一用 fallback 实现 (效果等价)
        self.upsample = _FallbackSpatialUp(d2, d1)

        # ------- 7. Decoder Scale-1 -------
        self.decoder_block = _build_attn_block(
            attn_type=attn_type, dim=d1, num_heads=num_heads,
            cuboid_size=cuboid_size, shift_size=shift_size, strategy=strategy,
            ffn_drop=ffn_drop, attn_drop=attn_drop,
        )

        # ------- 8. Output Head -------
        # 逐像素线性投影到 in_channels (SEVIR 为 1)
        self.output_head = nn.Linear(d1, in_channels)

    # ------------------------------------------------------------------ #
    # 核心: 循环瓶颈
    # ------------------------------------------------------------------ #
    def _recurrent_bottleneck(self, z: torch.Tensor, num_steps: int) -> torch.Tensor:
        """在同一个 attention block 中循环迭代 N 次 (权重共享)."""
        for _ in range(num_steps):
            if self.use_checkpoint and self.training:
                z = cp.checkpoint(self.recurrent_cell, z, use_reentrant=False)
            else:
                z = self.recurrent_cell(z)
        return z

    def _sample_num_steps(self, override: Optional[int] = None) -> int:
        if override is not None:
            return int(override)
        if self.training:
            lo, hi = self.num_steps_range
            return random.randint(lo, hi)
        return int(self.eval_num_steps)

    # ------------------------------------------------------------------ #
    # 前向
    # ------------------------------------------------------------------ #
    def forward(
        self,
        x: torch.Tensor,
        num_steps: Optional[int] = None,
        return_aux: bool = False,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : [B, T_in, H, W]  或  [B, T_in, H, W, C_in]
        num_steps : int, 可选; 覆盖训练期随机采样/推理期默认值
        return_aux : bool; 若 True 额外返回 (Z_0, Z_N) 便于调试梯度/正则项

        Returns
        -------
        y : [B, T_out, H, W]  (若输入是 4D)
            或 [B, T_out, H, W, C_in] (若输入是 5D)
        """
        squeeze_c = False
        if x.dim() == 4:
            x = x.unsqueeze(-1)  # [B,T,H,W] → [B,T,H,W,1]
            squeeze_c = True
        assert x.dim() == 5, f"输入维度必须为 4 或 5, 当前 {x.dim()}"
        B, T_in, H, W, C_in = x.shape
        assert T_in == self.in_frames, \
            f"输入帧数不匹配: got {T_in}, expected {self.in_frames}"
        assert C_in == self.in_channels, \
            f"输入通道不匹配: got {C_in}, expected {self.in_channels}"

        # 1. Patch embed: 通道 C_in → d1
        h = self.patch_embed(x)                             # [B,T_in,H,W,d1]

        # 2. Encoder Scale-1
        h = self.encoder_block(h)                           # [B,T_in,H,W,d1]

        # 3. 时间投影 T_in → T_out (在 Scale-1 分辨率上)
        #    permute 让 T 变为最后一维再做 Linear
        skip_1 = h.permute(0, 2, 3, 4, 1)                   # [B,H,W,d1,T_in]
        skip_1 = self.temporal_proj(skip_1)                 # [B,H,W,d1,T_out]
        skip_1 = skip_1.permute(0, 4, 1, 2, 3).contiguous() # [B,T_out,H,W,d1]

        # 4. 空间下采样 → Z_0
        z_0 = self.downsample(skip_1)                       # [B,T_out,H/2,W/2,d2]

        # 5. 循环瓶颈 (核心!)
        n_steps = self._sample_num_steps(num_steps)
        z_n = self._recurrent_bottleneck(z_0, n_steps)      # [B,T_out,H/2,W/2,d2]

        # 6. 空间上采样 (同时通道减半)
        u = self.upsample(z_n)                              # [B,T_out,H,W,d1]

        # 7. 跳跃连接融合 (相加, 形状一致)
        assert u.shape == skip_1.shape, \
            f"skip 尺寸不匹配: up={tuple(u.shape)}, skip={tuple(skip_1.shape)}"
        u = u + skip_1

        # 8. Decoder Scale-1
        u = self.decoder_block(u)                           # [B,T_out,H,W,d1]

        # 9. Output head
        y = self.output_head(u)                             # [B,T_out,H,W,C_in]

        if squeeze_c:
            y = y.squeeze(-1)                               # [B,T_out,H,W]

        if return_aux:
            return y, {"z_0": z_0, "z_n": z_n, "num_steps": n_steps}
        return y
