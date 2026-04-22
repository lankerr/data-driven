"""
重庆数据 · 4 个实验统一训练脚本
--------------------------------
用法:
    python train_chongqing.py --config configs_chongqing/e1_3h_to_3h.yaml
    python train_chongqing.py --config configs_chongqing/e2_21h_to_3h.yaml
    python train_chongqing.py --config configs_chongqing/e3_mstc_nodetach.yaml
    python train_chongqing.py --config configs_chongqing/e4_mstc_detach.yaml

支持:
  - AMP 混合精度 (training.amp: true)
  - 实验 tag 作为 save_dir / log_dir 子目录
  - 普通模型 (vit_unet 等) 与 MSTC 多尺度级联模型共用同一训练循环
"""
import os
import sys
import io
# 防止 Windows GBK 控制台遇到 ✓/✗ 等 Unicode 字符时崩溃
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
import copy
import time
import json
import argparse
import traceback

import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.chongqing_dataset import build_chongqing_loaders
from models import build_model
from models.multiscale_cascade import MultiScaleTemporalCascade, MSTCLoss
from utils.losses import build_loss
from utils.metrics import compute_all_metrics, print_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    return p.parse_args()


def load_cfg(path, args):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    return cfg


# --------------------------- Model & Loss build ---------------------------
def build_cq_model(cfg):
    exp = cfg["model"]["experiment"]
    if exp == "mstc":
        m_cfg = cfg.get("mstc", {})
        return MultiScaleTemporalCascade(
            use_fine=m_cfg.get("use_fine", True),
            detach_between_stages=m_cfg.get("detach_between_stages", False),
            vit_backbone=cfg["model"].get("vit_backbone", "swin_t"),
            pretrained=cfg["model"].get("pretrained", True),
            reduce_stem=cfg["model"].get("reduce_stem", True),
        )
    return build_model(cfg)


def build_cq_loss(cfg):
    base = build_loss(cfg)
    if cfg["model"]["experiment"] == "mstc":
        lw = cfg.get("mstc", {}).get("loss_weights", {})
        return MSTCLoss(
            base_loss=base,
            w_l0=lw.get("w_l0", 0.1),
            w_l1=lw.get("w_l1", 0.2),
            w_l2=lw.get("w_l2", 0.3),
            w_l3=lw.get("w_l3", 1.0),
        ), True
    return base, False


# --------------------------- Train / Validate ---------------------------
def forward_once(model, x, is_mstc):
    if is_mstc:
        return model(x)  # dict
    return model(x)      # tensor


def compute_loss(criterion, out, target, is_mstc):
    return criterion(out, target)


def get_pred_tensor(out, is_mstc):
    return out["final"] if is_mstc else out


def train_epoch(model, loader, criterion, optimizer, scaler, device,
                is_mstc, use_amp):
    model.train()
    total = 0.0
    n = 0
    pbar = tqdm(loader, desc="训练", leave=False)
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad()
        with autocast(enabled=use_amp):
            out = forward_once(model, x, is_mstc)
            loss = compute_loss(criterion, out, y, is_mstc)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total += loss.item()
        n += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total / max(n, 1)


@torch.no_grad()
def validate(model, loader, criterion, device, is_mstc, use_amp):
    model.eval()
    total = 0.0
    metrics_sum = {}
    n = 0
    for x, y in tqdm(loader, desc="验证", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast(enabled=use_amp):
            out = forward_once(model, x, is_mstc)
            loss = compute_loss(criterion, out, y, is_mstc)
        pred = get_pred_tensor(out, is_mstc).float()
        total += loss.item()
        m = compute_all_metrics(pred.cpu(), y.cpu())
        for k, v in m.items():
            metrics_sum[k] = metrics_sum.get(k, 0.0) + v
        n += 1
    avg_metrics = {k: v / n for k, v in metrics_sum.items()}
    return total / max(n, 1), avg_metrics


# --------------------------- Main ---------------------------
def main():
    args = parse_args()
    cfg = load_cfg(args.config, args)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg["training"].get("amp", False)) and device.startswith("cuda")
    tag = cfg.get("experiment_tag", os.path.splitext(os.path.basename(args.config))[0])

    print(f"{'='*60}\n  [{tag}] 启动训练\n{'='*60}")
    print(f"  device={device}, amp={use_amp}")
    print(f"  config: {args.config}")

    save_dir = os.path.join(cfg["training"]["save_dir"], tag)
    log_dir = os.path.join(cfg["training"]["log_dir"], tag)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 数据
    train_loader, val_loader, test_loader = build_chongqing_loaders(cfg)

    # 模型 + 损失
    model = build_cq_model(cfg).to(device)
    criterion, is_mstc = build_cq_loss(cfg)
    if isinstance(criterion, nn.Module):
        criterion.to(device)

    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  模型参数: total={total_p:,}  trainable={train_p:,}  "
          f"({'MSTC' if is_mstc else cfg['model']['experiment']})")

    # 优化器
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=cfg["training"]["lr"],
                      weight_decay=cfg["training"]["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"],
                                  eta_min=1e-7)
    scaler = GradScaler(enabled=use_amp)
    writer = SummaryWriter(log_dir)

    best_val = float("inf")
    patience_cnt = 0
    patience = cfg["training"].get("patience", 15)

    try:
        for epoch in range(1, cfg["training"]["epochs"] + 1):
            t0 = time.time()
            tr_loss = train_epoch(model, train_loader, criterion,
                                  optimizer, scaler, device, is_mstc, use_amp)
            val_loss, val_metrics = validate(model, val_loader, criterion,
                                             device, is_mstc, use_amp)
            scheduler.step()
            lr = optimizer.param_groups[0]["lr"]
            print(f"  E{epoch:3d} | tr {tr_loss:.4f} | val {val_loss:.4f} "
                  f"| CSI {val_metrics.get('CSI_avg', 0):.4f} "
                  f"| lr {lr:.2e} | {time.time()-t0:.1f}s")

            writer.add_scalar("loss/train", tr_loss, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            writer.add_scalar("lr", lr, epoch)
            for k, v in val_metrics.items():
                writer.add_scalar(f"metric/{k}", v, epoch)

            if val_loss < best_val - cfg["training"].get("min_delta", 1e-4):
                best_val = val_loss
                patience_cnt = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "val_metrics": val_metrics,
                    "config": cfg,
                }, os.path.join(save_dir, "best.pt"))
                print(f"    ✓ 保存最优 (val {val_loss:.4f})")
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    print(f"    ⚠ 早停 (patience={patience})")
                    break

        # 测试
        ckpt = torch.load(os.path.join(save_dir, "best.pt"), weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        test_loss, test_metrics = validate(model, test_loader, criterion,
                                           device, is_mstc, use_amp)
        print(f"\n  === TEST [{tag}] ===")
        print_metrics(test_metrics, prefix="    ")
        print(f"    test_loss={test_loss:.4f}")

        with open(os.path.join(save_dir, "result.json"), "w", encoding="utf-8") as f:
            json.dump({
                "tag": tag,
                "best_val_loss": best_val,
                "test_loss": test_loss,
                "test_metrics": test_metrics,
            }, f, indent=2, ensure_ascii=False)

    except torch.cuda.OutOfMemoryError as e:
        print(f"\n  ✗ OOM 错误: {e}")
        with open(os.path.join(save_dir, "OOM.flag"), "w", encoding="utf-8") as f:
            f.write(str(e))
        sys.exit(2)
    except Exception as e:
        print(f"\n  ✗ 训练异常: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        writer.close()


if __name__ == "__main__":
    main()
