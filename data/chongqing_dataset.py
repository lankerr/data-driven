"""
重庆雷达 VIL 数据集加载器
---------------------------------
- 目录: C:/Users/97290/Desktop/datasets/2026chongqing/vil_gpu_daily_240_simple
- 每个 npy: (240, 384, 384) float32, 值域 [0,1], 间隔 6min (240帧 = 24h)
- 排除 2024 年数据 (用户说有问题)
- 按日期排序后 train/val/test = 7/1.5/1.5

在同一 __getitem__ 中返回 (in_frames, out_frames) 的连续窗口，
滑窗仅在单日内部进行 (跨零点不跨)
"""
import os
import re
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


DATE_RE = re.compile(r"day_simple_(\d{8})\.npy$")


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
                 max_days=None):
        super().__init__()
        self.in_frames = in_frames
        self.out_frames = out_frames
        self.seq_len = in_frames + out_frames
        self.img_size = img_size

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
        seq = np.asarray(data[st:st + self.seq_len], dtype=np.float32).copy()  # [0,1]
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
