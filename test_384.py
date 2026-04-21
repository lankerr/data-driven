"""测试: 使用真实 SEVIR Lite 数据集的完整训练流程"""
import torch
from models import build_model, RESIDUAL_MODELS
from data.sevir_dataset import build_dataloaders
import yaml

def main():
    with open('config.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    print(f"img_size={cfg['data']['img_size']}, in={cfg['data']['in_frames']}, out={cfg['data']['out_frames']}")
    print(f"lite_dir={cfg['data']['lite_dir']}")

    # 测试时用 0 workers 避免 Windows spawn 问题
    cfg['data']['num_workers'] = 0

    # 1. 测试数据加载
    print("\n--- 测试数据加载 ---")
    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    x, y = next(iter(train_loader))
    print(f"input: {x.shape}, target: {y.shape}, dtype: {x.dtype}")
    print(f"value range: [{x.min():.3f}, {x.max():.3f}]")

    # 2. 测试所有模型
    print("\n--- 测试模型前向传播 ---")
    experiments = ['resnet_unet', 'vit_unet', 'resnet_vit_residual', 'vit_resnet_residual', 'hybrid_unet']

    for exp in experiments:
        cfg['model']['experiment'] = exp
        model = build_model(cfg)
        total = sum(p.numel() for p in model.parameters())

        with torch.no_grad():
            if exp in RESIDUAL_MODELS:
                y_final, y_base = model(x)
                print(f"{exp:>25s} | params: {total:>12,} | in: {tuple(x.shape)} -> out: {tuple(y_final.shape)}, base: {tuple(y_base.shape)}")
            else:
                pred = model(x)
                print(f"{exp:>25s} | params: {total:>12,} | in: {tuple(x.shape)} -> out: {tuple(pred.shape)}")

        del model

    print("\nAll OK!")

if __name__ == '__main__':
    main()
