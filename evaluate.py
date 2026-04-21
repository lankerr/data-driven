"""
评估脚本: 加载训练好的模型, 在测试集上计算所有指标
用法:
  python evaluate.py --checkpoint checkpoints/resnet_unet/best.pt
  python evaluate.py --compare checkpoints/  # 对比所有实验结果
"""
import os
import argparse
import yaml
import torch
from tqdm import tqdm
from tabulate import tabulate

from data.sevir_dataset import build_dataloaders
from models import build_model, RESIDUAL_MODELS
from utils.metrics import compute_all_metrics, print_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="评估模型")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="单个 checkpoint 路径")
    parser.add_argument("--compare", type=str, default="checkpoints",
                        help="对比目录下所有实验")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


@torch.no_grad()
def evaluate_model(model, loader, device, is_residual=False):
    """完整测试集评估"""
    model.eval()
    all_metrics = {}
    n_batches = 0

    # 逐帧指标
    frame_metrics = {}

    for batch in tqdm(loader, desc="评估"):
        if isinstance(batch[0], dict):
            inputs = {k: v.to(device) for k, v in batch[0].items()}
            target = batch[1].to(device)
        else:
            inputs = batch[0].to(device)
            target = batch[1].to(device)

        if is_residual:
            pred, _ = model(inputs)
        else:
            pred = model(inputs)

        # 整体指标
        metrics = compute_all_metrics(pred.cpu(), target.cpu())
        for k, v in metrics.items():
            all_metrics[k] = all_metrics.get(k, 0) + v

        # 逐帧 CSI (分析预测衰减)
        T = pred.shape[1]
        for t in range(T):
            frame_m = compute_all_metrics(pred[:, t:t+1].cpu(),
                                           target[:, t:t+1].cpu())
            for k, v in frame_m.items():
                key = f"frame_{t}_{k}"
                frame_metrics[key] = frame_metrics.get(key, 0) + v

        n_batches += 1

    avg = {k: v / n_batches for k, v in all_metrics.items()}
    frame_avg = {k: v / n_batches for k, v in frame_metrics.items()}

    return avg, frame_avg


def evaluate_checkpoint(ckpt_path, device):
    """加载 checkpoint 并评估"""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    _, _, test_loader = build_dataloaders(cfg)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    exp_name = cfg["model"]["experiment"]
    is_residual = exp_name in RESIDUAL_MODELS

    print(f"\n评估: {exp_name}")
    metrics, frame_metrics = evaluate_model(model, test_loader, device,
                                             is_residual)
    print_metrics(metrics)

    return exp_name, metrics, frame_metrics


def compare_experiments(base_dir, device):
    """对比所有实验结果"""
    results = {}

    for exp_dir in sorted(os.listdir(base_dir)):
        ckpt_path = os.path.join(base_dir, exp_dir, "best.pt")
        if not os.path.exists(ckpt_path):
            # 尝试加载保存的测试结果
            test_path = os.path.join(base_dir, exp_dir, "test_results.pt")
            if os.path.exists(test_path):
                data = torch.load(test_path, weights_only=False)
                results[exp_dir] = data["test_metrics"]
            continue

        try:
            name, metrics, _ = evaluate_checkpoint(ckpt_path, device)
            results[name] = metrics
        except Exception as e:
            print(f"跳过 {exp_dir}: {e}")

    if not results:
        print("未找到可评估的模型")
        return

    # 打印对比表
    print(f"\n{'='*80}")
    print("  实验对比总表")
    print(f"{'='*80}")

    # 表头
    key_metrics = ["MSE", "MAE", "CSI_avg", "CSI_轻度", "CSI_中度",
                   "CSI_重度", "CSI_极端"]

    headers = ["实验"] + key_metrics
    rows = []
    for exp_name, m in results.items():
        row = [exp_name]
        for km in key_metrics:
            row.append(f"{m.get(km, 0):.4f}")
        rows.append(row)

    # 简单表格输出
    print(f"\n{'实验':>25s}", end="")
    for km in key_metrics:
        print(f" | {km:>10s}", end="")
    print()
    print("-" * (25 + len(key_metrics) * 13))
    for row in rows:
        print(f"{row[0]:>25s}", end="")
        for v in row[1:]:
            print(f" | {v:>10s}", end="")
        print()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.checkpoint:
        evaluate_checkpoint(args.checkpoint, device)
    else:
        compare_experiments(args.compare, device)


if __name__ == "__main__":
    main()
