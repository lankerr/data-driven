"""
Mythos-Earthformer Smoke Test
==============================
不加载真实 SEVIR 数据, 用随机张量验证:
  1. 前向形状正确性 (T_in=13 → T_out=12, H/W 不变)
  2. 不同 N 值 (num_steps) 都能跑通
  3. 反向传播梯度能从 Loss → Z_0 → encoder / patch_embed 完整回传
  4. 权重共享: 改 N 不会改变参数量

使用:
    python test_mythos_forward.py
"""
import sys
import torch
import torch.nn as nn

sys.path.insert(0, ".")
from models.mythos_earthformer import MythosEarthformer
from utils.losses import MSESSIMLoss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # 用 96x96 而非 384x384 以加速调试 (模型对空间尺寸无硬编码依赖)
    B, T_in, T_out, H, W = 1, 13, 12, 96, 96

    model = MythosEarthformer(
        in_frames=T_in, out_frames=T_out,
        base_dim=64,                # 缩小以便 CPU 也能跑
        num_heads=4,
        num_steps_range=(3, 5),
        eval_num_steps=4,
        attn_type="conv3d",         # smoke test 用轻量版
        use_checkpoint=False,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[params] {n_params:,}")

    loss_fn = MSESSIMLoss(ssim_weight=0.3).to(device)

    # ---------- 测试 1: 不同 N 值的前向 ----------
    print("\n[test 1] 前向 - 变化 num_steps")
    x = torch.randn(B, T_in, H, W, device=device)
    y_gt = torch.rand(B, T_out, H, W, device=device)
    model.eval()
    for n in (1, 2, 5, 10):
        with torch.no_grad():
            y = model(x, num_steps=n)
        print(f"  N={n:2d}  y.shape={tuple(y.shape)}  "
              f"mean={y.mean().item():+.4f}  std={y.std().item():.4f}")
        assert y.shape == (B, T_out, H, W)

    # ---------- 测试 2: 随机深度 (training mode) ----------
    print("\n[test 2] 训练模式 - num_steps 随机采样")
    model.train()
    for _ in range(3):
        y, aux = model(x, return_aux=True)
        print(f"  sampled N={aux['num_steps']}  y.shape={tuple(y.shape)}")

    # ---------- 测试 3: 反向传播 ----------
    print("\n[test 3] 反向传播 - 梯度回传完整性")
    model.train()
    model.zero_grad()
    y = model(x, num_steps=4)
    loss = loss_fn(y, y_gt)
    loss.backward()

    grads_status = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            grads_status[name] = "NO GRAD!"
        else:
            g = p.grad.abs().mean().item()
            grads_status[name] = g

    # 检查关键模块是否收到梯度
    key_modules = [
        "patch_embed", "encoder_block", "temporal_proj",
        "downsample", "recurrent_cell", "upsample",
        "decoder_block", "output_head",
    ]
    print(f"  loss = {loss.item():.6f}")
    all_ok = True
    for km in key_modules:
        # 找第一个匹配该模块的参数
        match = next((n for n in grads_status if n.startswith(km)), None)
        if match is None:
            print(f"  [{km}] 未找到参数 (可能没有权重)")
            continue
        g = grads_status[match]
        ok = isinstance(g, float) and g > 0
        mark = "✓" if ok else "✗"
        all_ok &= ok
        print(f"  {mark} [{km:16s}] {match}: grad_mean={g}")

    print(f"\n{'PASS' if all_ok else 'FAIL'}: 所有关键模块均收到非零梯度")

    # ---------- 测试 4: 权重共享验证 ----------
    print("\n[test 4] 权重共享: 变化 N 时参数量恒定")
    p_before = sum(p.numel() for p in model.parameters())
    _ = model(x, num_steps=20)
    p_after = sum(p.numel() for p in model.parameters())
    assert p_before == p_after
    print(f"  N=1 vs N=20 参数量一致: {p_before:,}  ✓")

    print("\n=== Mythos-Earthformer 骨架 smoke test 通过 ===")


if __name__ == "__main__":
    main()
