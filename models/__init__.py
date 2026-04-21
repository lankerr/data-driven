"""
模型工厂: 根据配置创建对应的实验模型
"""
from .single_model import ResNetUNet, ViTUNet
from .residual_model import ResNetViTResidual, ViTResNetResidual
from .hybrid_model import HybridUNet
from .multimodal import EarlyFusionModel, MidFusionModel, LateFusionModel
from .mythos_earthformer import MythosEarthformer

MODEL_REGISTRY = {
    "resnet_unet":          ResNetUNet,
    "vit_unet":             ViTUNet,
    "resnet_vit_residual":  ResNetViTResidual,
    "vit_resnet_residual":  ViTResNetResidual,
    "hybrid_unet":          HybridUNet,
    "early_fusion":         EarlyFusionModel,
    "mid_fusion":           MidFusionModel,
    "late_fusion":          LateFusionModel,
    "mythos_earthformer":   MythosEarthformer,
}

# 残差模型需要特殊的损失函数处理
RESIDUAL_MODELS = {"resnet_vit_residual", "vit_resnet_residual"}


def build_model(cfg):
    """根据配置创建模型"""
    exp = cfg["model"]["experiment"]
    if exp not in MODEL_REGISTRY:
        raise ValueError(f"未知实验类型: {exp}, 可选: {list(MODEL_REGISTRY.keys())}")

    model_cls = MODEL_REGISTRY[exp]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    common = dict(
        in_channels=data_cfg["in_frames"],
        out_channels=data_cfg["out_frames"],
        pretrained=model_cfg.get("pretrained", True),
        reduce_stem=model_cfg.get("reduce_stem", False),
    )

    if exp in ("resnet_unet",):
        return model_cls(backbone=model_cfg["resnet_backbone"], **common)
    elif exp in ("vit_unet",):
        return model_cls(backbone=model_cfg["vit_backbone"], **common)
    elif exp in RESIDUAL_MODELS:
        return model_cls(
            resnet_backbone=model_cfg["resnet_backbone"],
            vit_backbone=model_cfg["vit_backbone"],
            **common,
        )
    elif exp == "hybrid_unet":
        return model_cls(
            resnet_backbone=model_cfg["resnet_backbone"],
            vit_backbone=model_cfg["vit_backbone"],
            **common,
        )
    elif exp in ("early_fusion", "mid_fusion", "late_fusion"):
        return model_cls(
            modalities=tuple(data_cfg["modalities"]),
            in_frames=data_cfg["in_frames"],
            out_frames=data_cfg["out_frames"],
            backbone=model_cfg["resnet_backbone"],
        )
    elif exp == "mythos_earthformer":
        myth_cfg = cfg.get("mythos", {}) or {}
        return model_cls(
            in_frames=data_cfg["in_frames"],
            out_frames=data_cfg["out_frames"],
            base_dim=myth_cfg.get("base_dim", 128),
            num_heads=myth_cfg.get("num_heads", 4),
            num_steps_range=tuple(myth_cfg.get("num_steps_range", [3, 8])),
            eval_num_steps=myth_cfg.get("eval_num_steps", 6),
            attn_type=myth_cfg.get("attn_type", "conv3d"),
            use_checkpoint=myth_cfg.get("use_checkpoint", False),
            in_channels=1,
            ffn_drop=model_cfg.get("dropout", 0.1),
        )
    else:
        return model_cls(**common)
