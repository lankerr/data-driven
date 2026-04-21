"""
运行全部 5 个实验, 训练 + 测试 + 对比
支持断点续训:
  - 如果 checkpoints/<exp>/done.flag 存在, 跳过该实验
  - 如果 checkpoints/<exp>/best.pt 存在但无 done.flag, 从它续训
用法: python run_all.py
"""
import os
import sys
import time
import copy
import json
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from data.sevir_dataset import build_dataloaders
from models import build_model, RESIDUAL_MODELS
from utils.losses import build_loss, ResidualLoss
from utils.metrics import compute_all_metrics


# ============================================================
#  配置
# ============================================================
EXPERIMENTS = [
    "resnet_unet",
    "vit_unet",
    "resnet_vit_residual",
    "vit_resnet_residual",
    "hybrid_unet",
]
EPOCHS = 100         # 最大训练轮数, 依靠早停结束
EVAL_EVERY = 1       # 每轮都验证
PATIENCE = 10        # 连续10轮无改善则早停


# ============================================================
#  训练一个 epoch
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, device, is_residual):
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        if is_residual:
            y_final, y_base = model(x)
            loss = criterion(y_final, y_base, y)
        else:
            pred = model(x)
            loss = criterion(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


# ============================================================
#  验证 / 测试
# ============================================================
@torch.no_grad()
def evaluate(model, loader, criterion, device, is_residual):
    model.eval()
    total_loss = 0.0
    all_metrics = {}
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if is_residual:
            y_final, y_base = model(x)
            loss = criterion(y_final, y_base, y)
            pred = y_final
        else:
            pred = model(x)
            loss = criterion(pred, y)
        total_loss += loss.item()
        metrics = compute_all_metrics(pred.cpu(), y.cpu())
        for k, v in metrics.items():
            all_metrics[k] = all_metrics.get(k, 0) + v
        n += 1
    avg_loss = total_loss / max(n, 1)
    avg_metrics = {k: v / n for k, v in all_metrics.items()}
    return avg_loss, avg_metrics


# ============================================================
#  运行单个实验
# ============================================================
def run_single(exp_name, cfg, train_loader, val_loader, test_loader, device):
    print(f"\n{'='*65}")
    print(f"  实验: {exp_name}")
    print(f"{'='*65}")

    cfg["model"]["experiment"] = exp_name
    model = build_model(cfg).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {params:,}")

    is_residual = exp_name in RESIDUAL_MODELS

    # 损失
    base_crit = build_loss(cfg)
    if is_residual:
        criterion = ResidualLoss(base_crit, copy.deepcopy(base_crit))
    else:
        criterion = base_crit

    optimizer = AdamW(model.parameters(), lr=cfg["training"]["lr"],
                      weight_decay=cfg["training"]["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)

    save_dir = os.path.join("checkpoints", exp_name)
    os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float("inf")
    patience_cnt = 0
    history = []

    # 断点续训: 如果存在 best.pt, 加载权重作为起点
    best_path = os.path.join(save_dir, "best.pt")
    if os.path.exists(best_path):
        print(f"  ➤ 发现旧 checkpoint, 从它续训")
        model.load_state_dict(torch.load(best_path, weights_only=True))
        # 用当前模型的 val_loss 作为基准
        with torch.no_grad():
            base_val_loss, _ = evaluate(model, val_loader, criterion,
                                        device, is_residual)
        best_val_loss = base_val_loss
        print(f"  ➤ 旧 best val_loss = {best_val_loss:.4f}")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion,
                                     optimizer, device, is_residual)
        scheduler.step()
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        # 验证
        if epoch % EVAL_EVERY == 0 or epoch == EPOCHS:
            val_loss, val_m = evaluate(model, val_loader, criterion,
                                       device, is_residual)
            csi = val_m.get("CSI_avg", 0)
            print(f"  Epoch {epoch:3d}/{EPOCHS} | "
                  f"Train {train_loss:.4f} | Val {val_loss:.4f} | "
                  f"CSI_avg {csi:.4f} | LR {lr:.1e} | {elapsed:.0f}s")
            history.append((epoch, train_loss, val_loss, csi))

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                patience_cnt = 0
                torch.save(model.state_dict(),
                           os.path.join(save_dir, "best.pt"))
            else:
                patience_cnt += 1
                if patience_cnt >= PATIENCE:
                    print(f"  ⚠ 早停 (epoch {epoch})")
                    break
        else:
            print(f"  Epoch {epoch:3d}/{EPOCHS} | "
                  f"Train {train_loss:.4f} | LR {lr:.1e} | {elapsed:.0f}s")

    # 加载最优并测试
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, weights_only=True))
    test_loss, test_m = evaluate(model, test_loader, criterion,
                                  device, is_residual)
    print(f"\n  --- 测试结果 ---")
    print(f"  Test Loss: {test_loss:.4f}")
    for k, v in test_m.items():
        print(f"    {k:>12s}: {v:.4f}")

    result = {"test_loss": test_loss, "params": params, **test_m}

    # 标记为完成 + 保存单实验结果
    with open(os.path.join(save_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    with open(os.path.join(save_dir, "done.flag"), "w") as f:
        f.write("1")

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return result


# ============================================================
#  主函数
# ============================================================
def main():
    # 加载配置
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Windows 下用 0 workers 避免 spawn 问题
    cfg["data"]["num_workers"] = 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")
    print(f"数据: {cfg['data'].get('lite_dir', 'synthetic')}")
    print(f"输入: {cfg['data']['in_frames']}帧, 输出: {cfg['data']['out_frames']}帧, "
          f"分辨率: {cfg['data']['img_size']}x{cfg['data']['img_size']}")
    print(f"训练轮数: {EPOCHS}, batch_size: {cfg['training']['batch_size']}")

    # 构建数据 (所有实验共享)
    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    # 逐个运行实验 (自动跳过已完成的)
    all_results = {}
    for exp in EXPERIMENTS:
        done_flag = os.path.join("checkpoints", exp, "done.flag")
        result_json = os.path.join("checkpoints", exp, "result.json")
        if os.path.exists(done_flag) and os.path.exists(result_json):
            print(f"\n[SKIP] {exp} 已完成, 跳过 (删除 {done_flag} 可重新训练)")
            with open(result_json, "r", encoding="utf-8") as f:
                all_results[exp] = json.load(f)
            continue
        all_results[exp] = run_single(
            exp, copy.deepcopy(cfg),
            train_loader, val_loader, test_loader, device
        )

    # ============================================================
    #  汇总对比表
    # ============================================================
    print(f"\n{'='*90}")
    print(f"  全部实验对比汇总")
    print(f"{'='*90}")

    header = (f"{'实验名':>25s} | {'参数量':>10s} | {'Test Loss':>10s} | "
              f"{'MSE':>8s} | {'MAE':>8s} | {'CSI_avg':>8s} | "
              f"{'CSI_轻度':>8s} | {'CSI_中度':>8s} | {'CSI_重度':>8s}")
    print(header)
    print("-" * 120)

    for exp, m in all_results.items():
        p = f"{m['params']/1e6:.1f}M"
        print(f"{exp:>25s} | {p:>10s} | {m['test_loss']:10.4f} | "
              f"{m['MSE']:8.4f} | {m['MAE']:8.4f} | {m['CSI_avg']:8.4f} | "
              f"{m.get('CSI_轻度', 0):8.4f} | {m.get('CSI_中度', 0):8.4f} | "
              f"{m.get('CSI_重度', 0):8.4f}")

    # 找最优
    best_exp = min(all_results, key=lambda k: all_results[k]["test_loss"])
    best_csi = max(all_results, key=lambda k: all_results[k]["CSI_avg"])
    print(f"\n  最低 Test Loss: {best_exp} ({all_results[best_exp]['test_loss']:.4f})")
    print(f"  最高 CSI_avg:   {best_csi} ({all_results[best_csi]['CSI_avg']:.4f})")

    # 保存结果
    torch.save(all_results, "all_results.pt")
    print(f"\n  结果已保存至 all_results.pt")


if __name__ == "__main__":
    main()
