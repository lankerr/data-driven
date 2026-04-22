"""
重庆 VIL 像素值分位数分析 (找合适的 B-MSE 阈值)
"""
import os, sys, glob, re
import numpy as np

if sys.platform == "win32":
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except: pass

ROOT = r"C:\Users\97290\Desktop\datasets\2026chongqing\vil_gpu_daily_240_simple"

files = sorted(glob.glob(os.path.join(ROOT, "day_simple_*.npy")))
files = [f for f in files if not re.search(r"day_simple_2024\d{4}\.npy$", f)]

# 抽 30 天得到较大样本 (~4亿像素)
sample = files[::len(files)//30][:30]
print(f"抽样 {len(sample)} 天")

arrs = []
for f in sample:
    a = np.load(f, mmap_mode="r")
    arrs.append(np.asarray(a).ravel())
all_pix = np.concatenate(arrs)
print(f"总像素: {len(all_pix)/1e6:.0f}M, min={all_pix.min():.4f} max={all_pix.max():.4f} mean={all_pix.mean():.4f}")

# 只看 >0 的部分
nonzero = all_pix[all_pix > 0.01]
print(f">0.01 占比: {len(nonzero)/len(all_pix)*100:.3f}%")
print(f">0.01 部分: min={nonzero.min():.4f} max={nonzero.max():.4f} mean={nonzero.mean():.4f}")

# 分位数
qs = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999, 0.9995, 0.9999]
print("\n全体像素分位数:")
for q in qs:
    v = np.quantile(all_pix, q)
    print(f"  p{q*100:<7.3f} = {v:.5f}")

print("\n>0.01 部分分位数:")
for q in qs:
    v = np.quantile(nonzero, q)
    print(f"  p{q*100:<7.3f} = {v:.5f}")

# 各阈值下占比
print("\n候选阈值占比 (归一化后):")
candidates = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
for t in candidates:
    p = (all_pix >= t).mean() * 100
    print(f"  >= {t:.3f} : {p:.5f}%")

# 推荐: 让 6 档占比分别约 [~99%, 0.5%, 0.2%, 0.1%, 0.05%, 0.01%]
print("\n推荐阈值 (基于分位数 p99.0, p99.5, p99.8, p99.9, p99.99):")
picks = [0.99, 0.995, 0.998, 0.999, 0.9999]
ts = [np.quantile(all_pix, q) for q in picks]
for q, t in zip(picks, ts):
    p = (all_pix >= t).mean() * 100
    print(f"  p{q*100:.3f}  t={t:.5f}   实际 >= : {p:.4f}%")

# 计算 6 档占比
edges = [-np.inf] + ts + [np.inf]
counts, _ = np.histogram(all_pix, bins=edges)
frac = counts / counts.sum()
print("\n基于推荐阈值的 6 档分布:")
names = ["无雨", "轻", "中", "重", "极端", "暴"]
for n, f in zip(names, frac):
    print(f"  {n:<8} {f*100:>8.4f}%")

# 合理权重 (按 1/sqrt(p) 缩放, 避免爆炸)
w = 1.0 / np.sqrt(np.maximum(frac, 1e-5))
w = w / w[0]
print("\n按 1/sqrt(频率) 归一化权重:")
for n, wi in zip(names, w):
    print(f"  {n:<8} {wi:>10.2f}")

# 温和: 1/(freq^0.4)
w2 = 1.0 / np.maximum(frac, 1e-5)**0.4
w2 = w2 / w2[0]
print("\n按 1/freq^0.4 (温和版) 权重:")
for n, wi in zip(names, w2):
    print(f"  {n:<8} {wi:>10.2f}")
