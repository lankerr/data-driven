"""
对已完成的实验: 加载 best.pt → 测试 → 写 result.json + done.flag
用法: python mark_done.py resnet_unet vit_unet
"""
import os, sys, json, copy, yaml, torch
from data.sevir_dataset import build_dataloaders
from models import build_model, RESIDUAL_MODELS
from utils.losses import build_loss, ResidualLoss
from utils.metrics import compute_all_metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device, is_residual):
    model.eval()
    total_loss, all_metrics, n = 0.0, {}, 0
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
        m = compute_all_metrics(pred.cpu(), y.cpu())
        for k, v in m.items():
            all_metrics[k] = all_metrics.get(k, 0) + v
        n += 1
    return total_loss / max(n, 1), {k: v/n for k, v in all_metrics.items()}


def main():
    if len(sys.argv) < 2:
        print("用法: python mark_done.py <exp1> [exp2] ...")
        sys.exit(1)

    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["data"]["num_workers"] = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, test_loader = build_dataloaders(cfg)

    for exp in sys.argv[1:]:
        save_dir = os.path.join("checkpoints", exp)
        best_path = os.path.join(save_dir, "best.pt")
        if not os.path.exists(best_path):
            print(f"[WARN] {exp}: 无 best.pt, 跳过")
            continue

        print(f"\n[{exp}] 加载 best.pt 并测试...")
        cfg2 = copy.deepcopy(cfg)
        cfg2["model"]["experiment"] = exp
        model = build_model(cfg2).to(device)
        model.load_state_dict(torch.load(best_path, weights_only=True))
        params = sum(p.numel() for p in model.parameters())

        is_residual = exp in RESIDUAL_MODELS
        base_crit = build_loss(cfg2)
        criterion = ResidualLoss(base_crit, copy.deepcopy(base_crit)) if is_residual else base_crit

        test_loss, test_m = evaluate(model, test_loader, criterion, device, is_residual)
        result = {"test_loss": test_loss, "params": params, **test_m}
        print(f"  Test Loss: {test_loss:.4f}")
        for k, v in test_m.items():
            print(f"    {k:>12s}: {v:.4f}")

        with open(os.path.join(save_dir, "result.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        with open(os.path.join(save_dir, "done.flag"), "w") as f:
            f.write("1")
        print(f"  ✓ 已标记 {exp} 为完成")

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


if __name__ == "__main__":
    main()
