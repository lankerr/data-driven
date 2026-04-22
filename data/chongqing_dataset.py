"""
重庆雷达 VIL 数据集加载器
---------------------------------
- 目录: C:/Users/97290/Desktop/datasets/2026chongqing/vil_gpu_daily_240_simple
- 每个 npy: (240, 384, 384) float32, 值域 [0,1] (= VIL_kg_m2 / 80), 间隔 6min
- 排除 2024 年数据 (用户说有问题)
- 按日期排序后 train/val/test = 7/1.5/1.5

重要预处理管线 (来自 MOE/nowcastnet 经验, 见 CHONGQING_EXPERIMENTS.md):
    stored(0~1) × 80 → VIL (kg/m²)
    → 截断 VIL < cutoff (默认 1.0 kg/m²) → 杂波归零
    → 可选 QMap(CQ → SEVIR)            ← data/assets/cq_to_sevir_qmap.npy
    → log1p(VIL) / log1p(80)            → 归一化到 [0,1]
阈值用物理 VIL (例如 [1,5,10,20,30,50] kg/m²), 评估时也按同一管线映射。
"""
import os
import re
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


DATE_RE = re.compile(r"day_simple_(\d{8})\.npy$")

# ─────────────── VIL 物理尺度参数 ────────────────
VIL_MAX = 80.0                           # kg/m²
LOG1P_VIL_MAX = float(np.log1p(VIL_MAX)) # ≈ 4.394
DEFAULT_QMAP = os.path.join(os.path.dirname(__file__), "assets", "cq_to_sevir_qmap.npy")


def _load_qmap(path):
    if path is None or not os.path.isfile(path):
        return None
    t = np.load(path).astype(np.float32)
    # 保证列 0 递增
    if t.ndim != 2 or t.shape[1] != 2:
        raise ValueError(f"QMap 表形状异常: {t.shape}, 期望 (N,2)")
    return t


def stored_to_vil(seq_stored, scale=VIL_MAX):
    """stored [0,1] → 物理 VIL kg/m²"""
    return seq_stored * scale


def apply_cq_pipeline(seq_stored, qmap_table=None, cutoff=1.0, use_qmap=True):
    """
    CQ 数据预处理管线 (numpy float32):
        stored → VIL(kg/m²) → cutoff → QMap(CQ→SEVIR) → log1p 归一化
    返回 [0,1] 范围的 float32 tensor 数据.
    """
    vil = stored_to_vil(seq_stored.astype(np.float32, copy=False))
    # 截断杂波
    if cutoff > 0:
        vil = np.where(vil > cutoff, vil, 0.0)
    # 分位数映射
    if use_qmap and qmap_table is not None:
        mask = vil > 0
        if mask.any():
            vil_mapped = np.zeros_like(vil)
            vil_mapped[mask] = np.interp(vil[mask], qmap_table[:, 0], qmap_table[:, 1])
            vil = vil_mapped
    # log1p 归一化
    vil_norm = np.log1p(np.maximum(vil, 0.0)) / LOG1P_VIL_MAX
    return vil_norm.astype(np.float32)


def vil_phys_to_norm(vil_phys):
    """物理 VIL (kg/m²) → log1p 归一化空间 (用于把阈值换算到归一化域)"""
    return float(np.log1p(max(vil_phys, 0.0)) / LOG1P_VIL_MAX)


def _list_days(root, exclude_years=(2024,)):
    files = []
    for f in os.listdir(root):
        m = DATE_RE.match(f)
        if not m:
            continue
        date = m.group(1)
        year = int(date[:4])
        if year in exclude_years:
            continue
        files.append((date, os.path.join(root, f)))
    files.sort(key=lambda x: x[0])  # 按日期升序
    return files


class ChongqingDailyDataset(Dataset):
    """
    从每个日 npy (240, 384, 384) 中滑窗切出 (in_frames + out_frames) 样本
    所有样本都在同一天内部，不跨天
    """

    def __init__(self, data_dir, split="train",
                 in_frames=30, out_frames=30, img_size=384,
                 stride=None,
                 train_ratio=0.7, val_ratio=0.15,
                 exclude_years=(2024,),
                 max_days=None,
                 use_qmap=True, qmap_path=None, vil_cutoff=1.0):
        super().__init__()
        self.in_frames = in_frames
        self.out_frames = out_frames
        self.seq_len = in_frames + out_frames
        self.img_size = img_size
        self.use_qmap = use_qmap
        self.vil_cutoff = float(vil_cutoff)
        self.qmap_table = _load_qmap(qmap_path or DEFAULT_QMAP) if use_qmap else None
        if use_qmap and self.qmap_table is None:
            print(f"[Chongqing][warn] 要求 use_qmap=True 但未找到 qmap 表, 降级为不做映射")

        all_days = _list_days(data_dir, exclude_years=exclude_years)
        n = len(all_days)
        assert n > 0, f"未找到可用天 (root={data_dir}, exclude={exclude_years})"

        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        if split == "train":
            sel = all_days[:n_train]
        elif split == "val":
            sel = all_days[n_train:n_train + n_val]
        elif split == "test":
            sel = all_days[n_train + n_val:]
        else:
            raise ValueError(split)

        if max_days:
            sel = sel[:max_days]

        self.days = sel  # [(date_str, path), ...]
        self.stride = stride if stride is not None else max(1, out_frames)

        self.samples = []  # (day_idx, start_frame)
        for di, (_, fp) in enumerate(self.days):
            t_total = np.load(fp, mmap_mode="r").shape[0]
            for st in range(0, t_total - self.seq_len + 1, self.stride):
                self.samples.append((di, st))

        print(f"[Chongqing] {split}: {len(self.days)} 天, "
              f"{len(self.samples)} 样本 "
              f"(in={in_frames}, out={out_frames}, stride={self.stride})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        di, st = self.samples[idx]
        fp = self.days[di][1]
        data = np.load(fp, mmap_mode="r")
        seq = np.asarray(data[st:st + self.seq_len], dtype=np.float32).copy()  # stored [0,1]
        # 正确预处理: stored → VIL → cutoff → QMap → log1p
        seq = apply_cq_pipeline(seq,
                                qmap_table=self.qmap_table,
                                cutoff=self.vil_cutoff,
                                use_qmap=self.use_qmap)
        seq = torch.from_numpy(seq)

        H, W = seq.shape[-2:]
        if H != self.img_size or W != self.img_size:
            seq = F.interpolate(seq.unsqueeze(1),
                                size=self.img_size,
                                mode="bilinear", align_corners=False).squeeze(1)

        x = seq[:self.in_frames]
        y = seq[self.in_frames:]
        return x, y


def build_chongqing_loaders(cfg):
    dc = cfg["data"]
    bs = cfg["training"]["batch_size"]
    nw = dc.get("num_workers", 2)

    common = dict(
        data_dir=dc["chongqing_dir"],
        in_frames=dc["in_frames"],
        out_frames=dc["out_frames"],
        img_size=dc.get("img_size", 384),
        stride=dc.get("stride", None),
        train_ratio=dc.get("train_ratio", 0.7),
        val_ratio=dc.get("val_ratio", 0.15),
        exclude_years=tuple(dc.get("exclude_years", [2024])),
        max_days=dc.get("max_days", None),
        use_qmap=dc.get("use_qmap", True),
        qmap_path=dc.get("qmap_path", None),
        vil_cutoff=dc.get("vil_cutoff", 1.0),
    )
    train_ds = ChongqingDailyDataset(split="train", **common)
    val_ds = ChongqingDailyDataset(split="val", **common)
    test_ds = ChongqingDailyDataset(split="test", **common)

    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True,
                   num_workers=nw, pin_memory=True, drop_last=True),
        DataLoader(val_ds, batch_size=bs, shuffle=False,
                   num_workers=nw, pin_memory=True),
        DataLoader(test_ds, batch_size=bs, shuffle=False,
                   num_workers=nw, pin_memory=True),
    )
