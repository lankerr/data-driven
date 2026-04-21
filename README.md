# data-driven · SEVIR 气象回波外推 × Mythos-Earthformer

> **0–3 小时雷达回波外推**的数据驱动实验集合：在 SEVIR 数据集上对比 CNN / ViT / 残差堆叠 / 门控融合五组骨干组合，并进一步提出 **Mythos-Earthformer**——把 Amazon Earthformer 的 Cuboid-Attention U-Net 骨架里最底层的瓶颈层改造成 **权重共享的循环推理引擎 (Continuous Latent Space Reasoning / Recurrent Depth)**。

![framework](https://img.shields.io/badge/framework-PyTorch-ee4c2c) ![dataset](https://img.shields.io/badge/dataset-SEVIR-blue) ![status](https://img.shields.io/badge/status-in--progress-orange)

## 目录

- [1. 研究动机](#1-研究动机)
- [2. Mythos-Earthformer 核心思想](#2-mythos-earthformer-核心思想)
- [3. 完整架构蓝图（Mermaid）](#3-完整架构蓝图mermaid)
- [4. 五组 CNN/ViT 消融实验](#4-五组-cnnvit-消融实验)
- [5. 六条"可再往上堆"的路线](#5-六条可再往上堆的路线)
- [6. 损失函数切换](#6-损失函数切换)
- [7. 训练/推理使用方法](#7-训练推理使用方法)
- [8. 本地 OOM → 服务器迁移手册](#8-本地-oom--服务器迁移手册)
- [9. 项目结构](#9-项目结构)
- [10. 参考文献](#10-参考文献)

## 1. 研究动机

传统雷达外推方法（光流、自回归 RNN/ConvLSTM/PredRNN、甚至 Earthformer 那样的时空 Transformer）都有两个共同痛点：

1. **自回归误差累积**：`t → t+1 → t+2 → ...` 一步步推，误差指数放大。
2. **层数堆叠浪费参数**：多个独立权重的 block 各学各的，没强迫它们学一个统一的"动力学步进算子"。

近期两个源头给了我们启发：

- **Earthformer (NeurIPS 2022, Amazon)** —— Cuboid Attention 把 `[T,H,W,C]` 切块做稀疏注意力，是 SEVIR 基准的经典骨架。
- **Recurrent Depth / Continuous Latent Reasoning (2024–2025)** —— Geiping et al. *Scaling up Test-Time Compute with Latent Reasoning* (arXiv:2502.05171)、Hao et al. *Coconut* (arXiv:2412.06769) 等把**单个 transformer block 当 RNN cell** 在潜空间循环 N 次，训练期随机 N（Random Depth Training），推理期可任意加大来换质量。

把后者迁移到时空预测，天然契合：大气是一个连续时间动力系统 $\partial_t \mathbf{u} = F(\mathbf{u})$，"同一算子迭代 N 步"恰好对应"在潜空间里多跑几步欧拉/RK 积分"。

## 2. Mythos-Earthformer 核心思想

> 不在每层用不同的权重；把 U-Net 的**瓶颈层砍掉一半**，换成**一个**可循环 N 次的 Cuboid Attention Block。

```
[B,13,384,384]
   │ PatchEmbed (1 → 128)
   ▼
Encoder Scale-1  (Cuboid Attention × 1)
   │
   ├──► TemporalProj (T: 13 → 12)  ═══════════ Skip_1 缓存
   │
   ▼ PatchMerging3D  (128 → 256, H/W ÷ 2)
Z_0  [B,12,192,192,256]
   │
   ▼ ╔════════════════════════════════════════════╗
     ║  Recurrent Latent Bottleneck (权重共享!)   ║
     ║    for _ in range(N):                      ║
     ║        Z = recurrent_cell(Z)   ← 同一权重  ║
     ║  # training: N ∼ U{3..8}                   ║
     ║  # eval:     N=6 默认, 可拉大到 10-20      ║
     ╚════════════════════════════════════════════╝
Z_N
   │ SpatialUpsample ×2  (256 → 128)
   │  + Skip_1                  ⟵⟵ U-Net 跳跃连接（相加）
   ▼ Decoder Scale-1  (Cuboid Attention × 1)
   │ OutputHead (128 → 1)
   ▼
[B,12,384,384]
```

**为什么在 Scale-1 做 `T_in → T_out` 投影？** 因为瓶颈既要循环又要输出 12 帧；把 T 对齐放到 Skip_1 之前，Z_0 / Z_N / skip 全部 T=12，**循环单元的 input/output shape 完全一致**，才能做真正的权重共享迭代。

**实现**：[`models/mythos_earthformer.py`](models/mythos_earthformer.py)，支持 `attn_type="cuboid"` 使用官方算子，或 `attn_type="conv3d"` 轻量占位用于调试。

**Smoke test 已通过** ✅（[`test_mythos_forward.py`](test_mythos_forward.py)）：

```
[test 1] 不同 N 前向:   N=1/2/5/10 → (1,12,96,96) ✓
[test 2] 随机深度训练:   每 forward 采样 N ∈ [3,5] ✓
[test 3] 反向传播:       梯度从 output_head → ... → patch_embed 全非零 ✓
[test 4] 权重共享:       N=1 与 N=20 参数量完全一致 ✓
```

## 3. 完整架构蓝图（Mermaid）

完整 8 区块结构图（含所有现有实验 + Mythos + 未来堆叠路线 + 训练实操）见 [`ARCHITECTURES.mmd`](ARCHITECTURES.mmd)。

下面是 Mythos 循环瓶颈的简化预览：

```mermaid
flowchart TB
  classDef input fill:#e3f2fd,stroke:#1976d2
  classDef loop fill:#ffebee,stroke:#c62828,stroke-width:3px
  classDef out fill:#ede7f6,stroke:#4527a0
  X["x [B,13,H,W]"]:::input
  PE["PatchEmbed + Encoder Scale-1"]
  TP["TemporalProj 13→12"]
  SK(["Skip_1"])
  DW["PatchMerging3D (÷2, 128→256)"]
  Z0["Z_0 [B,12,H/2,W/2,256]"]:::loop
  CELL["recurrent_cell<br/>(shared weights)"]:::loop
  ZN["Z_N"]:::loop
  UP["Upsample + add Skip_1"]
  DEC["Decoder Scale-1 + Head"]
  Y["y_hat [B,12,H,W]"]:::out
  X --> PE --> TP --> SK --> DW --> Z0
  Z0 -.->|for i in range(N)| CELL
  CELL -.->|weight tying| CELL
  CELL -.-> ZN --> UP --> DEC --> Y
```

## 4. 五组 CNN/ViT 消融实验

| # | 实验 | 模型 | 直觉 |
|---|---|---|---|
| 1 | `resnet_unet` | ResNet34 + UNet | CNN 画局部纹理，边缘锐利但全局偏 |
| 2 | `vit_unet` | Swin-T + UNet | Transformer 全局依赖，位置准但细节糊 |
| 3 | `resnet_vit_residual` | ResNet 打底 → ViT 修残差 | CNN 画轮廓，Transformer 做全局修正 |
| 4 ⭐ | `vit_resnet_residual` | ViT 打底 → ResNet 修残差 | 先全局平流 → 再局部对流修补 |
| 5 | `hybrid_unet` | 双分支并行 + 门控融合 | 逐尺度自适应混合 CNN/Transformer 特征 |

多模态版本（`early_fusion` / `mid_fusion` / `late_fusion`）见 [`models/multimodal.py`](models/multimodal.py)。

## 5. 六条"可再往上堆"的路线

这是准备在服务器上继续做的方向（详见 [`ARCHITECTURES.mmd`](ARCHITECTURES.mmd) 的 `STACK` 子图）：

| 路线 | 思路 | 一句话价值 |
|---|---|---|
| **S1** | Mythos 当 `y_base`，ResNet 修残差 | Exp-4 "ViT→CNN 残差" 的时空升级版 |
| **S2** | `recurrent_cell` 内部并联 Swin + Conv3D + FeatureFusionGate | Exp-5 "门控融合" 的时空版 |
| **S3** | VIL/IR/VIS 各自 PatchEmbed → Scale-1 Cross-Attention 融合 → 共享同一个循环瓶颈 | 多模态 Mythos |
| **S4** | Mythos 的 `Z_N` 当 diffusion 的条件（接 PreDiff / DiffCast） | 治长程预测模糊（high-freq collapse） |
| **S5** | 推理期 N=4/8/16/32 各跑一次几何平均 | **0 额外训练成本** 的 ensemble |
| **S6** | 把 `recurrent_cell` 换成 `AttnResStackCuboidSelfAttentionBlock` | 零改动接入 SOTA 变体 |

## 6. 损失函数切换

| loss 名 | 公式 | 用途 |
|---|---|---|
| `mse` | MSE | baseline |
| `bmse` | 按降水阈值分级加权 MSE | 解决长尾 |
| `ssim` | 1 - SSIM | 结构保真 |
| `combined` (旧) | (1-α)·**B-MSE** + α·SSIM | 之前默认 |
| `mse_ssim` (⭐新) | (1-α)·**MSE** + α·SSIM | **与 Earthformer 官方对齐**，梯度更干净 |

在 [`config.yaml`](config.yaml) 中切换：

```yaml
training:
  loss: "mse_ssim"
  ssim_weight: 0.3
```

## 7. 训练/推理使用方法

### 7.1 安装

```bash
pip install -r requirements.txt
```

### 7.2 Smoke test（无需数据）

```bash
python test_mythos_forward.py
```

### 7.3 训练 Mythos-Earthformer

```yaml
# config.yaml 最小修改
model:
  experiment: "mythos_earthformer"
training:
  loss: "mse_ssim"
mythos:
  base_dim: 128
  num_heads: 4
  num_steps_range: [3, 8]   # 训练期 N 随机
  eval_num_steps: 6         # 推理期默认
  attn_type: "cuboid"       # 生产用; 调试可改 "conv3d"
  use_checkpoint: false     # 显存紧时开 true
```

```bash
python train.py --experiment mythos_earthformer
```

### 7.4 推理时"深度思考"

```python
pred = model(x, num_steps=20)   # 拉大 N 换质量
```

### 7.5 最该做的消融图：N vs CSI

训一个模型，推理枚举 `N=1,2,4,6,10,15,20` 画曲线——**这是论文的核心卖点图**。

## 8. 本地 OOM → 服务器迁移手册

本地 RTX 在 384×384 + Cuboid Attention 下即使 `batch_size=2` 也会 OOM。迁到服务器后按如下阶梯稳扎稳打：

| 阶段 | 分辨率 | base_dim | N 范围 | 其它 |
|---|---|---|---|---|
| ① 5 epoch 验证 | 96×96 | 64 | [3,5] | `attn_type="cuboid"` 确认 loss 下降 |
| ② 10 epoch | 192×192 | 128 | [3,6] | `bf16` + `use_checkpoint=true` |
| ③ 100 epoch 完整 | 384×384 | 128 | [3,8] | `micro_batch=1` + `grad_accum=16` |

推荐优化器 / 调度 / 精度（参考 Earthformer 官方）：

```yaml
optim:
  method: adamw
  lr: 1.0e-3
  weight_decay: 0.0
  gradient_clip_val: 1.0
  lr_scheduler_mode: cosine
  warmup_percentage: 0.2
trainer:
  precision: "16-mixed"    # 或 bf16-mixed
  accumulate_grad_batches: 16
```

## 9. 项目结构

```
data_driven/
├── config.yaml                  # 统一配置 (已加入 mythos: 区块)
├── train.py / evaluate.py       # 训练 / 评估
├── visualize.py                 # 可视化
├── test_mythos_forward.py       # Smoke test (✅ 已通过)
├── ARCHITECTURES.mmd            # 完整架构 Mermaid 图 (8 区块)
├── MYTHOS_EARTHFORMER.md        # 详细架构文档
├── data/
│   └── sevir_dataset.py         # SEVIR + SEVIR-Lite 加载器
├── models/
│   ├── __init__.py              # 模型工厂 (已注册 mythos_earthformer)
│   ├── components.py            # ResNet/Swin encoder + UNet decoder
│   ├── single_model.py          # Exp-1/2
│   ├── residual_model.py        # Exp-3/4
│   ├── hybrid_model.py          # Exp-5
│   ├── multimodal.py            # Early/Mid/Late fusion
│   └── mythos_earthformer.py    # ⭐ 新模型
└── utils/
    ├── losses.py                # B-MSE / SSIM / Combined / MSESSIMLoss(新)
    └── metrics.py               # CSI / POD / FAR
```

## 10. 参考文献

- Gao, Zhihan, et al. *Earthformer: Exploring Space-Time Transformers for Earth System Forecasting.* NeurIPS 2022.
- Geiping, Jonas, et al. *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach.* arXiv:2502.05171, 2025.
- Hao, Shibo, et al. *Training Large Language Models to Reason in a Continuous Latent Space.* arXiv:2412.06769, 2024.
- Veillette, Mark S., et al. *SEVIR: A Storm Event Imagery Dataset for Deep Learning Applications.* NeurIPS 2020.

---

仅供科研使用。
# SEVIR 气象回波预测实验

基于 SEVIR 数据集的纯数据驱动气象回波外推实验，探索 **CNN (ResNet)** 与 **Transformer (Swin)** 在时空序列预测中的最优结合方式。

## 实验设计

### 五组核心消融实验

| # | 实验 | 模型 | 直觉 |
|---|------|------|------|
| 1 | `resnet_unet` | ResNet34 编码器 + UNet 解码器 | CNN 捕捉局部纹理，边缘锐利但全局位置可能偏 |
| 2 | `vit_unet` | Swin-T 编码器 + UNet 解码器 | Transformer 建模全局依赖，位置准但细节模糊 |
| 3 | `resnet_vit_residual` | ResNet 打底 → ViT 修残差 | CNN画轮廓，Transformer补全局修正 |
| 4 | `vit_resnet_residual` | ViT 打底 → ResNet 修残差 ⭐ | **最推荐**：先全局平流预测→再局部对流修补 |
| 5 | `hybrid_unet` | 双分支门控融合 | 逐层自适应混合CNN和Transformer特征 |

### 多模态融合实验

| 融合策略 | 说明 |
|----------|------|
| `early_fusion` | 输入端拼接所有模态（VIL+可见光+水汽+红外） |
| `mid_fusion` | 独立编码器 → 交叉注意力融合 → 共享解码 |
| `late_fusion` | 独立模型预测 → 可学习加权集成 |

## 项目结构

```
data_driven/
├── config.yaml                # 统一配置文件
├── train.py                   # 训练脚本
├── evaluate.py                # 评估与对比脚本
├── visualize.py               # 可视化工具
├── data/
│   └── sevir_dataset.py       # SEVIR 数据集加载 (含合成数据调试模式)
├── models/
│   ├── __init__.py            # 模型工厂
│   ├── components.py          # 编码器/解码器基础组件
│   ├── single_model.py        # 实验 1 & 2
│   ├── residual_model.py      # 实验 3 & 4
│   ├── hybrid_model.py        # 实验 5
│   └── multimodal.py          # 多模态融合
└── utils/
    ├── losses.py              # B-MSE / SSIM / 组合损失函数
    └── metrics.py             # CSI / POD / FAR 评价指标
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 数据准备

下载 SEVIR 数据集到 `./data/sevir/` 目录：
- 数据集地址：https://sevir.mit.edu/
- 需要 `CATALOG.csv` 和对应模态的 HDF5 文件

**无数据也能运行**：代码自动检测，若无 SEVIR 数据则使用合成数据进行调试。

### 3. 训练单个实验

```bash
# 实验1: 纯 ResNet-UNet
python train.py --experiment resnet_unet

# 实验4: ViT打底 + ResNet修残差 (推荐)
python train.py --experiment vit_resnet_residual

# 实验5: 混合融合
python train.py --experiment hybrid_unet
```

### 4. 一键运行全部5个实验

```bash
python train.py --experiment all
```

### 5. 评估与对比

```bash
# 对比所有实验结果
python evaluate.py --compare checkpoints/

# 评估单个模型
python evaluate.py --checkpoint checkpoints/vit_resnet_residual/best.pt
```

### 6. 可视化

```bash
python visualize.py --checkpoint checkpoints/vit_resnet_residual/best.pt
```

## 关键技术决策

### Time-as-Channel

预训练模型（ResNet/Swin）接受3通道输入，SEVIR是13帧时序。通过 `1×1 Conv` 将13帧映射到3通道，让预训练权重生效：

```
输入 (B, 13, H, W) → 1×1 Conv → (B, 3, H, W) → 预训练编码器
```

### B-MSE 损失

气象回波严重长尾分布（大面积无雨、小面积暴雨）。B-MSE 对强降水区域赋予高权重：

| VIL 阈值 | 对应强度 | 损失权重 |
|-----------|----------|----------|
| < 16      | 无降水   | ×1       |
| 16-74     | 轻度     | ×2       |
| 74-133    | 中度     | ×5       |
| 133-160   | 重度     | ×10      |
| > 181     | 暴雨     | ×30      |

### 评价指标

- **CSI**（临界成功指数）：最核心的气象指标，综合考虑命中和误报
- **POD**（命中率）：模型捕捉到了多少真实的降水事件
- **FAR**（误报率）：模型预测的降水中有多少是假的

## 配置说明

所有超参数集中在 `config.yaml`，主要可调项：

```yaml
model:
  experiment: "vit_resnet_residual"  # 选择实验
  pretrained: true                    # 是否用 ImageNet 预训练
  freeze_encoder: false               # 是否冻结编码器

training:
  batch_size: 8
  lr: 1.0e-4
  loss: "combined"                    # mse / bmse / ssim / combined
  ssim_weight: 0.3

data:
  img_size: 128                       # 128 快速原型，384 完整分辨率
  modalities: ["vil"]                 # 多模态: ["vil", "vis", "ir069", "ir107"]
```
