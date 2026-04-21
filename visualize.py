"""
可视化工具
  - 预测 vs 真实对比图
  - 逐帧预测动画
  - 多实验 CSI 对比曲线
  - 预测误差热力图
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec


# 气象雷达回波颜色映射 (模拟标准 NWS 颜色)
RADAR_COLORS = [
    (1.0, 1.0, 1.0, 0.0),   # 透明 (无降水)
    (0.0, 0.9, 0.0, 1.0),   # 绿色 (弱)
    (0.0, 0.6, 0.0, 1.0),   # 深绿
    (1.0, 1.0, 0.0, 1.0),   # 黄色
    (1.0, 0.6, 0.0, 1.0),   # 橙色
    (1.0, 0.0, 0.0, 1.0),   # 红色
    (0.6, 0.0, 0.0, 1.0),   # 深红
    (0.8, 0.0, 0.8, 1.0),   # 紫色 (极端)
]
RADAR_CMAP = mcolors.LinearSegmentedColormap.from_list("radar", RADAR_COLORS, N=256)


def plot_prediction_comparison(inputs, target, prediction, save_path=None,
                                n_frames=6, title=""):
    """
    并排展示: 输入 / 真实值 / 预测值
    从 T 帧中均匀采样 n_frames 帧
    """
    T_in = inputs.shape[0]
    T_out = target.shape[0]

    # 采样帧索引
    in_idx = np.linspace(0, T_in - 1, min(n_frames, T_in), dtype=int)
    out_idx = np.linspace(0, T_out - 1, min(n_frames, T_out), dtype=int)

    n_cols = max(len(in_idx), len(out_idx))
    fig, axes = plt.subplots(3, n_cols, figsize=(3 * n_cols, 9))
    fig.suptitle(title or "预测对比", fontsize=14, fontweight="bold")

    row_labels = ["输入帧", "真实值", "预测值"]

    for col, t in enumerate(in_idx):
        axes[0, col].imshow(inputs[t].numpy(), cmap=RADAR_CMAP, vmin=0, vmax=1)
        axes[0, col].set_title(f"t-{T_in - t}", fontsize=9)
        axes[0, col].axis("off")

    for col, t in enumerate(out_idx):
        axes[1, col].imshow(target[t].numpy(), cmap=RADAR_CMAP, vmin=0, vmax=1)
        axes[1, col].set_title(f"t+{t+1}", fontsize=9)
        axes[1, col].axis("off")

        axes[2, col].imshow(prediction[t].numpy(), cmap=RADAR_CMAP,
                            vmin=0, vmax=1)
        axes[2, col].set_title(f"t+{t+1}", fontsize=9)
        axes[2, col].axis("off")

    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=12, fontweight="bold",
                                 rotation=90, labelpad=10)

    # 隐藏多余子图
    for row in range(3):
        for col in range(n_cols):
            if (row == 0 and col >= len(in_idx)) or \
               (row > 0 and col >= len(out_idx)):
                axes[row, col].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  图片已保存: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_error_heatmap(target, prediction, save_path=None):
    """预测误差热力图 (逐帧)"""
    T = target.shape[0]
    error = (prediction - target).numpy()

    n_cols = min(T, 6)
    idx = np.linspace(0, T - 1, n_cols, dtype=int)

    fig, axes = plt.subplots(1, n_cols, figsize=(3 * n_cols, 3))
    if n_cols == 1:
        axes = [axes]

    for i, t in enumerate(idx):
        im = axes[i].imshow(error[t], cmap="RdBu_r", vmin=-0.5, vmax=0.5)
        axes[i].set_title(f"t+{t+1}", fontsize=9)
        axes[i].axis("off")

    fig.colorbar(im, ax=axes, label="预测误差", shrink=0.8)
    fig.suptitle("预测误差热力图 (红=高估, 蓝=低估)", fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def plot_csi_comparison(results_dict, save_path=None):
    """
    多实验 CSI 对比柱状图
    results_dict: {experiment_name: metrics_dict}
    """
    threshold_names = ["轻度", "中度", "重度", "极端"]
    experiments = list(results_dict.keys())
    n_exp = len(experiments)
    n_thresh = len(threshold_names)

    x = np.arange(n_thresh)
    width = 0.8 / n_exp

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (exp_name, metrics) in enumerate(results_dict.items()):
        csi_vals = [metrics.get(f"CSI_{tn}", 0) for tn in threshold_names]
        offset = (i - n_exp / 2 + 0.5) * width
        bars = ax.bar(x + offset, csi_vals, width, label=exp_name)

    ax.set_xlabel("降水强度阈值", fontsize=12)
    ax.set_ylabel("CSI", fontsize=12)
    ax.set_title("各实验 CSI 对比", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(threshold_names)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def plot_frame_csi_decay(frame_metrics_dict, save_path=None):
    """
    逐帧 CSI 衰减曲线 (分析预测随时间的退化)
    frame_metrics_dict: {exp_name: frame_metrics}
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for exp_name, fm in frame_metrics_dict.items():
        # 提取各帧的 CSI_avg
        frames = sorted(set(int(k.split("_")[1]) for k in fm.keys()))
        csi_values = []
        for t in frames:
            key = f"frame_{t}_CSI_avg"
            if key in fm:
                csi_values.append(fm[key])

        if csi_values:
            minutes = [(t + 1) * 5 for t in frames[:len(csi_values)]]
            ax.plot(minutes, csi_values, "-o", label=exp_name, markersize=4)

    ax.set_xlabel("预测时间 (分钟)", fontsize=12)
    ax.set_ylabel("CSI (平均)", fontsize=12)
    ax.set_title("预测 CSI 随时间衰减曲线", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 65)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def visualize_from_checkpoint(ckpt_path, n_samples=5, save_dir="figures"):
    """从 checkpoint 加载模型并可视化预测结果"""
    from data.sevir_dataset import build_dataloaders
    from models import build_model, RESIDUAL_MODELS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    exp_name = cfg["model"]["experiment"]

    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _, _, test_loader = build_dataloaders(cfg)
    is_residual = exp_name in RESIDUAL_MODELS

    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= n_samples:
                break

            if isinstance(batch[0], dict):
                inputs = {k: v.to(device) for k, v in batch[0].items()}
                target = batch[1].to(device)
                vis_input = list(batch[0].values())[0][0].cpu()
            else:
                inputs = batch[0].to(device)
                target = batch[1].to(device)
                vis_input = batch[0][0].cpu()

            if is_residual:
                pred, _ = model(inputs)
            else:
                pred = model(inputs)

            # 取 batch 中第一个样本
            plot_prediction_comparison(
                vis_input,
                target[0].cpu(),
                pred[0].cpu(),
                save_path=os.path.join(save_dir, f"{exp_name}_sample_{i}.png"),
                title=f"{exp_name} - 样本 {i}",
            )

            plot_error_heatmap(
                target[0].cpu(),
                pred[0].cpu(),
                save_path=os.path.join(save_dir, f"{exp_name}_error_{i}.png"),
            )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--save_dir", type=str, default="figures")
    args = parser.parse_args()

    visualize_from_checkpoint(args.checkpoint, args.n_samples, args.save_dir)
