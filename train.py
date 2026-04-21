"""
统一训练脚本
用法:
  python train.py                          # 使用默认配置
  python train.py --experiment resnet_unet  # 指定实验
  python train.py --config my_config.yaml   # 自定义配置文件
  python train.py --experiment all          # 顺序运行所有5个实验
"""
import os
import sys
import argparse
import time
import yaml
import copy
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.sevir_dataset import build_dataloaders
from models import build_model, RESIDUAL_MODELS
from utils.losses import build_loss, ResidualLoss
from utils.metrics import compute_all_metrics, print_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="SEVIR 气象回波预测实验")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--experiment", type=str, default=None,
                        help="实验类型或 'all' 运行全部")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None,
                        help="从 checkpoint 恢复训练")
    return parser.parse_args()


def load_config(args):
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 命令行参数覆盖
    if args.experiment:
        cfg["model"]["experiment"] = args.experiment
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr:
        cfg["training"]["lr"] = args.lr

    return cfg


def build_optimizer(model, cfg):
    train_cfg = cfg["training"]
    return AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )


def build_scheduler(optimizer, cfg):
    train_cfg = cfg["training"]
    epochs = train_cfg["epochs"]
    sched_type = train_cfg.get("scheduler", "cosine")

    if sched_type == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    elif sched_type == "step":
        return StepLR(optimizer, step_size=30, gamma=0.5)
    elif sched_type == "plateau":
        return ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
    else:
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)


def train_one_epoch(model, loader, criterion, optimizer, device,
                    is_residual=False):
    model.train()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc="训练", leave=False)
    for batch in pbar:
        if isinstance(batch[0], dict):
            inputs = {k: v.to(device) for k, v in batch[0].items()}
            target = batch[1].to(device)
        else:
            inputs = batch[0].to(device)
            target = batch[1].to(device)

        optimizer.zero_grad()

        if is_residual:
            y_final, y_base = model(inputs)
            loss = criterion(y_final, y_base, target)
        else:
            pred = model(inputs)
            loss = criterion(pred, target)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, criterion, device, is_residual=False):
    model.eval()
    total_loss = 0.0
    all_metrics = {}
    n_batches = 0

    for batch in tqdm(loader, desc="验证", leave=False):
        if isinstance(batch[0], dict):
            inputs = {k: v.to(device) for k, v in batch[0].items()}
            target = batch[1].to(device)
        else:
            inputs = batch[0].to(device)
            target = batch[1].to(device)

        if is_residual:
            y_final, y_base = model(inputs)
            loss = criterion(y_final, y_base, target)
            pred = y_final
        else:
            pred = model(inputs)
            loss = criterion(pred, target)

        total_loss += loss.item()

        # 计算指标
        metrics = compute_all_metrics(pred.cpu(), target.cpu())
        for k, v in metrics.items():
            all_metrics[k] = all_metrics.get(k, 0) + v

        n_batches += 1

    # 平均
    avg_loss = total_loss / max(n_batches, 1)
    avg_metrics = {k: v / n_batches for k, v in all_metrics.items()}

    return avg_loss, avg_metrics


def run_experiment(cfg, device):
    """运行单个实验"""
    exp_name = cfg["model"]["experiment"]
    print(f"\n{'='*60}")
    print(f"  实验: {exp_name}")
    print(f"{'='*60}")

    # 数据
    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    # 模型
    model = build_model(cfg).to(device)
    total, trainable = sum(p.numel() for p in model.parameters()), \
                       sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  参数量: {total:,} (可训练: {trainable:,})")

    # 损失
    is_residual = exp_name in RESIDUAL_MODELS
    if is_residual:
        base_criterion = build_loss(cfg)
        criterion = ResidualLoss(base_criterion, copy.deepcopy(base_criterion))
    else:
        criterion = build_loss(cfg)

    # 优化器
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    # TensorBoard
    log_dir = os.path.join(cfg["training"]["log_dir"], exp_name)
    writer = SummaryWriter(log_dir)

    # 训练循环
    save_dir = os.path.join(cfg["training"]["save_dir"], exp_name)
    os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0
    patience = cfg["training"].get("patience", 15)

    epochs = cfg["training"]["epochs"]
    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion,
                                     optimizer, device, is_residual)
        val_loss, val_metrics = validate(model, val_loader, criterion,
                                          device, is_residual)

        # 学习率调度
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        print(f"  Epoch {epoch:3d}/{epochs} | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"CSI_avg: {val_metrics.get('CSI_avg', 0):.4f} | "
              f"LR: {lr:.2e} | {elapsed:.1f}s")

        # TensorBoard 记录
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("LR", lr, epoch)
        for k, v in val_metrics.items():
            writer.add_scalar(f"Metrics/{k}", v, epoch)

        # 保存最优模型
        if val_loss < best_val_loss - cfg["training"].get("min_delta", 1e-4):
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_metrics": val_metrics,
                "config": cfg,
            }, os.path.join(save_dir, "best.pt"))
            print(f"  ✓ 保存最优模型 (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  ⚠ 早停: {patience} 轮无改善")
                break

    writer.close()

    # 最终测试
    print(f"\n  --- 测试集评估 ---")
    ckpt = torch.load(os.path.join(save_dir, "best.pt"), weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_loss, test_metrics = validate(model, test_loader, criterion,
                                        device, is_residual)
    print_metrics(test_metrics, prefix="  ")
    print(f"  Test Loss: {test_loss:.4f}")

    # 保存测试结果
    torch.save({
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "config": cfg,
    }, os.path.join(save_dir, "test_results.pt"))

    return test_metrics


def main():
    args = parse_args()
    cfg = load_config(args)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    os.makedirs(cfg["training"]["save_dir"], exist_ok=True)
    os.makedirs(cfg["training"]["log_dir"], exist_ok=True)

    # 运行全部实验 或 单个实验
    if cfg["model"]["experiment"] == "all":
        all_experiments = [
            "resnet_unet",
            "vit_unet",
            "resnet_vit_residual",
            "vit_resnet_residual",
            "hybrid_unet",
        ]
        all_results = {}
        for exp in all_experiments:
            exp_cfg = copy.deepcopy(cfg)
            exp_cfg["model"]["experiment"] = exp
            all_results[exp] = run_experiment(exp_cfg, device)

        # 打印对比表
        print(f"\n{'='*70}")
        print("  所有实验对比")
        print(f"{'='*70}")
        header = f"{'实验':>25s} | {'MSE':>8s} | {'CSI_avg':>8s} | {'CSI_重度':>8s}"
        print(header)
        print("-" * 60)
        for exp, m in all_results.items():
            print(f"{exp:>25s} | {m['MSE']:8.4f} | "
                  f"{m['CSI_avg']:8.4f} | {m.get('CSI_重度', 0):8.4f}")
    else:
        run_experiment(cfg, device)


if __name__ == "__main__":
    main()
