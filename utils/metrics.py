"""
评价指标
  - MSE / MAE: 基础回归指标
  - SSIM: 结构相似性
  - CSI / POD / FAR: 基于阈值的分类指标 (气象核心指标)
"""
import torch
import numpy as np


def compute_mse(pred, target):
    """均方误差"""
    return ((pred - target) ** 2).mean().item()


def compute_mae(pred, target):
    """平均绝对误差"""
    return (pred - target).abs().mean().item()


def compute_csi(pred, target, threshold):
    """
    Critical Success Index (临界成功指数)
    CSI = TP / (TP + FP + FN)
    最核心的气象预测评价指标
    """
    pred_bin = (pred >= threshold).float()
    tgt_bin = (target >= threshold).float()

    tp = (pred_bin * tgt_bin).sum().item()
    fp = (pred_bin * (1 - tgt_bin)).sum().item()
    fn = ((1 - pred_bin) * tgt_bin).sum().item()

    if tp + fp + fn == 0:
        return 1.0  # 都是无雨, 完美预测
    return tp / (tp + fp + fn)


def compute_pod(pred, target, threshold):
    """
    Probability of Detection (命中率)
    POD = TP / (TP + FN)
    """
    pred_bin = (pred >= threshold).float()
    tgt_bin = (target >= threshold).float()

    tp = (pred_bin * tgt_bin).sum().item()
    fn = ((1 - pred_bin) * tgt_bin).sum().item()

    if tp + fn == 0:
        return 1.0
    return tp / (tp + fn)


def compute_far(pred, target, threshold):
    """
    False Alarm Rate (误报率)
    FAR = FP / (TP + FP)
    """
    pred_bin = (pred >= threshold).float()
    tgt_bin = (target >= threshold).float()

    tp = (pred_bin * tgt_bin).sum().item()
    fp = (pred_bin * (1 - tgt_bin)).sum().item()

    if tp + fp == 0:
        return 0.0
    return fp / (tp + fp)


def compute_all_metrics(pred, target, thresholds=None, threshold_names=None):
    """
    计算所有指标, 返回结构化字典

    pred, target: (B, T, H, W), 值域 [0, 1]
    thresholds: 归一化后的 VIL 阈值列表
    """
    if thresholds is None:
        thresholds = [16/255, 74/255, 133/255, 160/255, 181/255]
    if threshold_names is None:
        threshold_names = ["轻度", "中度", "重度", "极端", "暴雨"]

    results = {
        "MSE": compute_mse(pred, target),
        "MAE": compute_mae(pred, target),
    }

    # 按阈值计算 CSI / POD / FAR
    for thresh, name in zip(thresholds, threshold_names):
        results[f"CSI_{name}"] = compute_csi(pred, target, thresh)
        results[f"POD_{name}"] = compute_pod(pred, target, thresh)
        results[f"FAR_{name}"] = compute_far(pred, target, thresh)

    # 各阈值 CSI 的平均 (综合指标)
    csi_values = [results[f"CSI_{n}"] for n in threshold_names]
    results["CSI_avg"] = np.mean(csi_values)

    return results


def print_metrics(metrics, prefix=""):
    """格式化打印指标"""
    header = f"{'指标':>12s}  {'值':>8s}"
    print(f"\n{prefix}{header}")
    print(f"{prefix}{'-' * 22}")
    for key, val in metrics.items():
        print(f"{prefix}{key:>12s}  {val:8.4f}")
