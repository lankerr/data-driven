"""
用新的 CQ 管线 (×80 + cutoff + QMap + log1p) 重算分布和阈值
"""
import os, sys, glob, re
import numpy as np

if sys.platform == "win32":
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except: pass

sys.path.insert(0, os.path.dirname(__file__))
from data.chongqing_dataset import (
    apply_cq_pipeline, _load_qmap, DEFAULT_QMAP, LOG1P_VIL_MAX, vil_phys_to_norm, VIL_MAX
)

ROOT = r"C:\Users\97290\Desktop\datasets\2026chongqing\vil_gpu_daily_240_simple"

qmap = _load_qmap(DEFAULT_QMAP)
print(f"QMap 表: {qmap.shape}, CQ 域 [{qmap[0,0]:.3f}, {qmap[-1,0]:.3f}], SEVIR 域 [{qmap[0,1]:.3f}, {qmap[-1,1]:.3f}]")

# 抽 30 天
files = sorted(glob.glob(os.path.join(ROOT, "day_simple_*.npy")))
files = [f for f in files if not re.search(r"day_simple_2024\d{4}\.npy$", f)]
sample = files[::len(files)//30][:30]
print(f"抽样 {len(sample)} 天")

# 3 种管线: raw (目前代码) / +cutoff_only / +cutoff+QMap
variants = [
    ("A. 原始 stored", dict(use_qmap=False, cutoff=0.0)),
    ("B. ×80 → cutoff=1 → log1p", dict(use_qmap=False, cutoff=1.0)),
    ("C. ×80 → cutoff=1 → QMap → log1p (推荐)", dict(use_qmap=True, cutoff=1.0)),
]
agg = {k: [] for k, _ in variants}
for i, f in enumerate(sample):
    a = np.load(f, mmap_mode="r")
    stored = np.asarray(a).astype(np.float32)  # (240, H, W)
    # 抽 20 帧
    idxs = np.random.RandomState(0+i).choice(stored.shape[0], 20, replace=False)
    frames = stored[idxs]
    for name, opts in variants:
        if name.startswith("A"):
            agg[name].append(frames.ravel())
        else:
            normed = apply_cq_pipeline(frames, qmap_table=qmap, **opts)
            agg[name].append(normed.ravel())

THRESH_VIL_PHYS = [1, 5, 10, 20, 30, 50]
print()
for name, _ in variants:
    arr = np.concatenate(agg[name])
    nz = arr[arr > 0]
    print("="*72)
    print(name)
    print(f"  min={arr.min():.5f}  max={arr.max():.5f}  mean={arr.mean():.5f}")
    print(f"  非零占比: {len(nz)/len(arr)*100:.4f}%")
    if len(nz) > 0:
        print(f"  非零: min={nz.min():.5f}  max={nz.max():.5f}  mean={nz.mean():.5f}")
    for q in [0.99, 0.995, 0.999, 0.9999]:
        print(f"    p{q*100:.3f}: {np.quantile(arr, q):.5f}")
    # 检查 VIL 物理阈值对应的归一化值和占比
    if name.startswith("C"):
        print("  物理阈值 → log1p 归一化 → 实际占比 (整体):")
        for v in THRESH_VIL_PHYS:
            # QMap 映射后仍是 kg/m², 所以阈值也走 QMap
            v_mapped = float(np.interp(v, qmap[:,0], qmap[:,1]))
            t_norm = vil_phys_to_norm(v_mapped)
            pct = (arr >= t_norm).mean() * 100
            print(f"    VIL={v:<3d} → mapped={v_mapped:.3f} → norm={t_norm:.4f} → 占比 {pct:.4f}%")
    elif name.startswith("B"):
        print("  物理阈值 → log1p → 占比 (整体):")
        for v in THRESH_VIL_PHYS:
            t_norm = vil_phys_to_norm(v)
            pct = (arr >= t_norm).mean() * 100
            print(f"    VIL={v:<3d} → norm={t_norm:.4f} → 占比 {pct:.4f}%")
