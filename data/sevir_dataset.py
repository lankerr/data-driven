"""
SEVIR 数据集加载器
支持 VIL / 可见光 / 水汽 / 红外 / 闪电 多模态数据
"""
import os
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


# SEVIR 各模态对应的 HDF5 key 和归一化系数
MODAL_INFO = {
    "vil":   {"key": "vil",   "scale": 1.0 / 255.0, "channels": 1},
    "vis":   {"key": "vis",   "scale": 1.0 / 255.0, "channels": 1},
    "ir069": {"key": "ir069", "scale": 1.0 / 255.0, "channels": 1},
    "ir107": {"key": "ir107", "scale": 1.0 / 255.0, "channels": 1},
    "lght":  {"key": "lght",  "scale": 1.0,         "channels": 1},
}


class SEVIRDataset(Dataset):
    """
    SEVIR 数据集
    将连续帧切分为 (输入帧, 目标帧) 对
    支持多模态输入, 以 VIL 为预测目标
    """

    def __init__(self, catalog_path, data_dir, modalities=("vil",),
                 in_frames=13, out_frames=13, img_size=128,
                 split="train", train_ratio=0.7, val_ratio=0.15):
        """
        Args:
            catalog_path: CATALOG.csv 路径
            data_dir: SEVIR 数据文件目录
            modalities: 使用的模态列表
            in_frames: 输入帧数
            out_frames: 输出帧数
            img_size: 目标空间分辨率
            split: train / val / test
            train_ratio / val_ratio: 数据集划分比例
        """
        super().__init__()
        self.data_dir = data_dir
        self.modalities = list(modalities)
        self.in_frames = in_frames
        self.out_frames = out_frames
        self.total_frames = in_frames + out_frames
        self.img_size = img_size

        # 加载并过滤 catalog
        self.catalog = self._load_catalog(catalog_path, split,
                                          train_ratio, val_ratio)
        # 构建样本索引: (event_idx, start_frame)
        self.samples = self._build_samples()

    def _load_catalog(self, catalog_path, split, train_ratio, val_ratio):
        """加载 CATALOG.csv 并按时间排序划分数据集"""
        catalog = pd.read_csv(catalog_path, parse_dates=["time_utc"])

        # 只保留包含 VIL 的事件 (VIL是必须的预测目标)
        vil_events = catalog[catalog["img_type"] == "vil"]
        event_ids = vil_events["id"].unique()

        # 如果使用多模态, 过滤出所有模态都存在的事件
        if len(self.modalities) > 1:
            for modal in self.modalities:
                modal_events = catalog[catalog["img_type"] == modal]["id"].unique()
                event_ids = np.intersect1d(event_ids, modal_events)

        # 按时间排序
        event_ids = sorted(event_ids)
        n = len(event_ids)

        # 划分
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        if split == "train":
            selected = event_ids[:n_train]
        elif split == "val":
            selected = event_ids[n_train:n_train + n_val]
        else:
            selected = event_ids[n_train + n_val:]

        selected = set(selected)
        return catalog[catalog["id"].isin(selected)]

    def _build_samples(self):
        """构建 (事件ID, 起始帧) 的样本列表"""
        samples = []
        vil_events = self.catalog[self.catalog["img_type"] == "vil"]

        for _, row in vil_events.iterrows():
            event_id = row["id"]
            file_name = row["file_name"]
            file_index = row["file_index"]
            n_frames = row.get("episode_length", 49)  # VIL 默认49帧

            # 滑动窗口
            for start in range(0, n_frames - self.total_frames + 1,
                               self.in_frames):
                samples.append({
                    "event_id": event_id,
                    "file_name": file_name,
                    "file_index": file_index,
                    "start_frame": start,
                })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        start = sample["start_frame"]
        end = start + self.total_frames

        inputs = {}
        target = None

        for modal in self.modalities:
            data = self._load_modal(sample, modal, start, end)
            if data is None:
                # 模态缺失时用零填充
                data = torch.zeros(self.total_frames, self.img_size, self.img_size)
            inputs[modal] = data[:self.in_frames]  # (T_in, H, W)

            if modal == "vil":
                target = data[self.in_frames:]  # (T_out, H, W)

        # 如果只有单模态 VIL, 直接返回张量
        if len(self.modalities) == 1 and "vil" in self.modalities:
            return inputs["vil"], target  # (T_in, H, W), (T_out, H, W)

        # 多模态: 返回字典
        return inputs, target

    def _load_modal(self, sample, modal, start, end):
        """从 HDF5 加载指定模态的数据"""
        # 查找该事件对应模态的文件
        event_rows = self.catalog[
            (self.catalog["id"] == sample["event_id"]) &
            (self.catalog["img_type"] == modal)
        ]

        if len(event_rows) == 0:
            return None

        row = event_rows.iloc[0]
        file_path = os.path.join(self.data_dir, modal, row["file_name"])

        if not os.path.exists(file_path):
            return None

        info = MODAL_INFO[modal]

        with h5py.File(file_path, "r") as f:
            key = info["key"]
            if key not in f:
                return None

            file_idx = row["file_index"]
            # SEVIR 数据格式: (N_events, L, H, W)
            data = f[key][file_idx, start:end]  # (T, H, W)

        # 转 float 并归一化
        data = torch.from_numpy(data.astype(np.float32)) * info["scale"]

        # 调整空间分辨率
        if data.shape[-1] != self.img_size:
            data = data.unsqueeze(1)  # (T, 1, H, W)
            data = F.interpolate(data, size=self.img_size,
                                 mode="bilinear", align_corners=False)
            data = data.squeeze(1)

        return data


class SEVIRLiteDataset(Dataset):
    """
    SEVIR Lite 数据集 — 加载预提取的 npy 文件
    每个 npy 文件形状: (T_total, H, W), dtype=uint8, 值域 [0, 255]
    目录结构: data_dir/{train,val,test}/*.npy
    """

    def __init__(self, data_dir, split="train",
                 in_frames=13, out_frames=12, img_size=128,
                 stride=None, **kwargs):
        super().__init__()
        self.in_frames = in_frames
        self.out_frames = out_frames
        self.seq_len = in_frames + out_frames
        self.img_size = img_size

        split_dir = os.path.join(data_dir, split)
        self.npy_files = sorted([
            os.path.join(split_dir, f)
            for f in os.listdir(split_dir) if f.endswith(".npy")
        ])
        print(f"[SEVIRLite] {split}: {len(self.npy_files)} 个事件, "
              f"in={in_frames}, out={out_frames}, seq={self.seq_len}")

        # 构建样本索引: (file_idx, start_frame)
        # 每个 npy 有 T_total 帧, 用滑动窗口切出多个 seq_len 长的样本
        if stride is None:
            stride = self.seq_len  # 默认不重叠
        self.samples = []
        for fi, fp in enumerate(self.npy_files):
            # 快速读 shape 而不加载数据
            t_total = np.load(fp, mmap_mode="r").shape[0]
            for start in range(0, t_total - self.seq_len + 1, stride):
                self.samples.append((fi, start))

        print(f"[SEVIRLite] {split}: 共 {len(self.samples)} 个样本 (stride={stride})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fi, start = self.samples[idx]
        # mmap 读取避免内存爆炸
        data = np.load(self.npy_files[fi], mmap_mode="r")
        seq = data[start:start + self.seq_len].copy()  # (seq_len, H, W), uint8

        # uint8 [0,255] → float32 [0,1]
        seq = torch.from_numpy(seq.astype(np.float32)) / 255.0

        # 如需 resize
        H, W = seq.shape[1], seq.shape[2]
        if H != self.img_size or W != self.img_size:
            seq = seq.unsqueeze(1)  # (T, 1, H, W)
            seq = F.interpolate(seq, size=self.img_size,
                                mode="bilinear", align_corners=False)
            seq = seq.squeeze(1)  # (T, H, W)

        x = seq[:self.in_frames]   # (in_frames, H, W)
        y = seq[self.in_frames:]   # (out_frames, H, W)
        return x, y


class SEVIRSyntheticDataset(Dataset):
    """
    合成数据集, 用于在没有真实 SEVIR 数据时调试代码
    生成随机的雷达回波样式数据
    """

    def __init__(self, num_samples=1000, modalities=("vil",),
                 in_frames=13, out_frames=13, img_size=128, **kwargs):
        self.num_samples = num_samples
        self.modalities = list(modalities)
        self.in_frames = in_frames
        self.out_frames = out_frames
        self.img_size = img_size
        self.total_frames = in_frames + out_frames

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """生成模拟移动回波"""
        torch.manual_seed(idx)
        T = self.total_frames
        H = W = self.img_size

        # 生成随机椭圆回波并模拟平移
        frames = torch.zeros(T, H, W)
        n_blobs = torch.randint(2, 6, (1,)).item()

        for _ in range(n_blobs):
            cx = torch.randint(20, W - 20, (1,)).item()
            cy = torch.randint(20, H - 20, (1,)).item()
            rx = torch.randint(5, 25, (1,)).item()
            ry = torch.randint(5, 25, (1,)).item()
            intensity = torch.rand(1).item() * 0.8 + 0.2
            vx = (torch.rand(1).item() - 0.5) * 3
            vy = (torch.rand(1).item() - 0.5) * 3

            yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W),
                                     indexing="ij")

            for t in range(T):
                cx_t = cx + vx * t
                cy_t = cy + vy * t
                mask = ((xx - cx_t) / rx) ** 2 + ((yy - cy_t) / ry) ** 2
                blob = intensity * torch.exp(-mask * 2)
                frames[t] += blob

        frames = frames.clamp(0, 1)

        if len(self.modalities) == 1:
            return frames[:self.in_frames], frames[self.in_frames:]

        inputs = {}
        for modal in self.modalities:
            noise = torch.randn_like(frames[:self.in_frames]) * 0.05
            inputs[modal] = (frames[:self.in_frames] + noise).clamp(0, 1)
        target = frames[self.in_frames:]
        return inputs, target


def build_dataloaders(cfg):
    """根据配置构建 train/val/test 数据加载器"""
    data_cfg = cfg["data"]
    modalities = data_cfg["modalities"]
    bs = cfg["training"]["batch_size"]
    nw = data_cfg.get("num_workers", 4)

    # 优先检查 SEVIR Lite npy 目录
    lite_dir = data_cfg.get("lite_dir", "")
    if lite_dir and os.path.isdir(os.path.join(lite_dir, "train")):
        print(f"[INFO] 使用 SEVIR Lite npy 数据: {lite_dir}")
        common = dict(
            data_dir=lite_dir,
            in_frames=data_cfg["in_frames"],
            out_frames=data_cfg["out_frames"],
            img_size=data_cfg["img_size"],
            stride=data_cfg.get("stride", None),
        )
        train_ds = SEVIRLiteDataset(split="train", **common)
        val_ds = SEVIRLiteDataset(split="val", **common)
        test_ds = SEVIRLiteDataset(split="test", **common)
    else:
        # 检查 SEVIR 原始 HDF5
        catalog = data_cfg.get("catalog_path", "")
        if os.path.exists(catalog):
            common_h5 = dict(
                modalities=modalities,
                in_frames=data_cfg["in_frames"],
                out_frames=data_cfg["out_frames"],
                img_size=data_cfg["img_size"],
            )
            base = dict(
                catalog_path=catalog,
                data_dir=data_cfg["sevir_dir"],
                train_ratio=data_cfg["train_ratio"],
                val_ratio=data_cfg["val_ratio"],
                **common_h5,
            )
            train_ds = SEVIRDataset(split="train", **base)
            val_ds = SEVIRDataset(split="val", **base)
            test_ds = SEVIRDataset(split="test", **base)
        else:
            print("[INFO] SEVIR 数据未找到, 使用合成数据进行调试")
            common_syn = dict(
                modalities=modalities,
                in_frames=data_cfg["in_frames"],
                out_frames=data_cfg["out_frames"],
                img_size=data_cfg["img_size"],
            )
            train_ds = SEVIRSyntheticDataset(num_samples=2000, **common_syn)
            val_ds = SEVIRSyntheticDataset(num_samples=400, **common_syn)
            test_ds = SEVIRSyntheticDataset(num_samples=400, **common_syn)

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=nw, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False,
                             num_workers=nw, pin_memory=True)

    return train_loader, val_loader, test_loader
