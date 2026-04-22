"""
重庆 vs SEVIR 分布对比
---------------------
在 VIL 归一化后 ([0,1]) 统计各像素值区间占比, 帮助判断 B-MSE 的
阈值/权重是否需要调整。

SEVIR 默认阈值: [16, 74, 133, 160, 181] / 255
对应 6 个区间, 默认权重 [1, 2, 5, 10, 30, 50]。

用法:
    python verify_distribution.py
"""
import os
import sys
import glob
import re
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

CHONGQING_ROOT = r"C:\Users\97290\Desktop\datasets\2026chongqing\vil_gpu_daily_240_simple"
SEVIR_CANDIDATES = [
    r"C:\Users\97290\Desktop\datasets\sevir",
    r"C:\Users\97290\Desktop\data_driven\data\sevir",
]

THRESHOLDS = [16/255, 74/255, 133/255, 160/255, 181/255]
BIN_NAMES = ["<16 (无雨)", "16-74 (轻)", "74-133 (中)", "133-160 (重)", "160-181 (极端)", ">=181 (暴)"]
DEFAULT_W = [1.0, 2.0, 5.0, 10.0, 30.0, 50.0]


def hist_from_array(arr_flat, thresholds=THRESHOLDS):
    """返回 6 个区间的像素占比 (fractions, sum=1)."""
    edges = np.array([-np.inf] + list(thresholds) + [np.inf])
    counts, _ = np.histogram(arr_flat, bins=edges)
    total = counts.sum()
    return counts.astype(np.float64) / max(total, 1), total


def analyze_chongqing(root, max_days=None):
    files = sorted(glob.glob(os.path.join(root, "day_simple_*.npy")))
    # 排除 2024
    files = [f for f in files if not re.search(r"day_simple_2024\d{4}\.npy$", f)]
    if max_days is not None:
        files = files[:max_days]
    print(f"[Chongqing] 扫描 {len(files)} 个日文件...")
    agg = np.zeros(6, dtype=np.float64)
    total_pix = 0
    for i, f in enumerate(files):
        a = np.load(f, mmap_mode="r").astype(np.float32).ravel()
        fr, tot = hist_from_array(a)
        agg += fr * tot
        total_pix += tot
        if (i+1) % 20 == 0 or i == len(files)-1:
            print(f"  进度 {i+1}/{len(files)}  累计像素 {total_pix/1e9:.2f}G")
    return agg / max(total_pix, 1), total_pix, len(files)


def analyze_sevir(root):
    """SEVIR 通常是 .h5 per catalog. 尝试快速抽样."""
    import h5py
    h5s = glob.glob(os.path.join(root, "**", "*.h5"), recursive=True)[:5]
    if not h5s:
        return None, 0, 0
    print(f"[SEVIR] 抽样 {len(h5s)} 个 h5 文件...")
    agg = np.zeros(6, dtype=np.float64)
    total_pix = 0
    n_samples = 0
    for f in h5s:
        try:
            with h5py.File(f, "r") as hf:
                # 找 vil 数据集
                vil_key = None
                for k in hf.keys():
                    if "vil" in k.lower():
                        vil_key = k; break
                if vil_key is None:
                    vil_key = list(hf.keys())[0]
                ds = hf[vil_key]
                # 随机抽 32 个事件, 归一化到 [0,1]
                n = min(32, ds.shape[0])
                idx = np.random.choice(ds.shape[0], n, replace=False)
                for i in sorted(idx):
                    a = np.asarray(ds[i]).astype(np.float32)
                    if a.max() > 1.5:
                        a = a / 255.0
                    fr, tot = hist_from_array(a.ravel())
                    agg += fr * tot
                    total_pix += tot
                    n_samples += 1
        except Exception as e:
            print(f"  [warn] {f}: {e}")
    return (agg / max(total_pix, 1)) if total_pix else None, total_pix, n_samples


def print_table(name, frac, total_pix):
    print(f"\n{'='*70}\n{name}  (总像素 {total_pix/1e9:.2f}G)\n{'='*70}")
    print(f"{'区间':<20} {'占比':>10} {'log10':>10} {'当前权重':>10} {'加权贡献':>12}")
    contribs = []
    for i in range(6):
        p = frac[i]
        w = DEFAULT_W[i]
        c = p * w
        contribs.append(c)
        lp = np.log10(max(p, 1e-20))
        print(f"{BIN_NAMES[i]:<20} {p*100:>9.4f}% {lp:>10.2f} {w:>10.1f} {c:>12.4e}")
    total_contrib = sum(contribs)
    print(f"{'-'*70}")
    print(f"加权贡献占比:")
    for i in range(6):
        print(f"  {BIN_NAMES[i]:<20} {contribs[i]/total_contrib*100:>8.2f}%")
    return np.array(contribs)


def suggest_weights(frac_cq, target_contrib=None):
    """
    目标: 让每个区间对 loss 的加权贡献 = target_contrib.
    target_contrib 默认给极端档更多权重: [0.02, 0.10, 0.20, 0.25, 0.23, 0.20]
    """
    if target_contrib is None:
        target_contrib = np.array([0.02, 0.10, 0.20, 0.25, 0.23, 0.20])
    # w_i * p_i  正比于 target_i  =>  w_i = target_i / p_i
    p = np.maximum(frac_cq, 1e-10)
    w = target_contrib / p
    # 归一到 w[0]=1
    w = w / w[0]
    return w


def main():
    print("### 分析重庆数据 ###")
    frac_cq, total_cq, n_days = analyze_chongqing(CHONGQING_ROOT)
    print_table(f"重庆 ({n_days} 天)", frac_cq, total_cq)

    frac_sv = None
    for root in SEVIR_CANDIDATES:
        if os.path.isdir(root):
            print(f"\n### 尝试分析 SEVIR @ {root} ###")
            frac_sv, total_sv, n_sv = analyze_sevir(root)
            if frac_sv is not None and total_sv > 0:
                print_table(f"SEVIR 抽样 ({n_sv} 个事件)", frac_sv, total_sv)
                break
    if frac_sv is None:
        print("\n[SEVIR] 未找到 h5 数据, 跳过对比")

    # 推荐权重
    print("\n" + "="*70 + "\n推荐的 B-MSE 权重 (基于重庆分布)\n" + "="*70)
    w_default = np.array(DEFAULT_W)
    w_sugg_balanced = suggest_weights(frac_cq, np.array([0.02, 0.10, 0.20, 0.25, 0.23, 0.20]))
    w_sugg_mild = suggest_weights(frac_cq, np.array([0.05, 0.15, 0.25, 0.25, 0.18, 0.12]))
    print(f"{'区间':<20} {'当前':>10} {'推荐-强':>12} {'推荐-温和':>12}")
    for i in range(6):
        print(f"{BIN_NAMES[i]:<20} {w_default[i]:>10.2f} {w_sugg_balanced[i]:>12.2f} {w_sugg_mild[i]:>12.2f}")

    print("\n解读:")
    print("  '推荐-强': 各档对 loss 的贡献约 [2%, 10%, 20%, 25%, 23%, 20%] (强监督稀有档)")
    print("  '推荐-温和': 各档贡献约 [5%, 15%, 25%, 25%, 18%, 12%] (较保守)")
    print("  如果重庆稀有档比 SEVIR 占比大很多 -> 权重可适当调小, 否则训练早期极端档贡献过大不稳")
    print("  如果重庆稀有档比 SEVIR 小很多 -> 权重需要调大, 否则模型仍然倾向全零")

    # 保存
    out = {
        "chongqing": {
            "fractions": frac_cq.tolist(),
            "bin_names": BIN_NAMES,
            "thresholds": THRESHOLDS,
            "total_pixels": int(total_cq),
            "n_days": n_days,
        },
        "weights_default": DEFAULT_W,
        "weights_suggested_strong": w_sugg_balanced.tolist(),
        "weights_suggested_mild": w_sugg_mild.tolist(),
    }
    if frac_sv is not None:
        out["sevir"] = {"fractions": frac_sv.tolist()}
    import json
    with open("distribution_report.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n报告已保存到 distribution_report.json")


if __name__ == "__main__":
    main()
