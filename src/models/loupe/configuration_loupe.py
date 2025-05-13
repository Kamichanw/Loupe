from typing import Literal, Optional
from omegaconf import DictConfig, ListConfig, OmegaConf
from transformers.configuration_utils import PretrainedConfig
from transformers.models.mask2former import Mask2FormerConfig
from loguru import logger

from models.pe import PE_VISION_CONFIG, PEConfig


def may_convert_to_object(obj):
    """Convert a ListConfig or DictConfig to a list or dict."""
    if isinstance(obj, (ListConfig, DictConfig)):
        return OmegaConf.to_container(obj, resolve=True)
    return obj


class LoupeConfig(PretrainedConfig):
    model_type = "loupe"

    def __init__(
        self,
        stage: Literal["cls", "seg", "cls_seg", "test"] = "cls",
        # basic configs
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        initializer_range=0.02,
        # backbone configs
        backbone_name: str = "PE-Core-L14-336",
        backbone_path: Optional[str] = None,
        backbone_overrides: Optional[dict] = None,
        # loupe configs - classification
        cls_mlp_ratio=2,  # 2 times of Perception Encoder hidden dim
        cls_mlp_layers=2,
        enable_patch_cls=True,
        enable_cls_fusion=True,
        freeze_backbone=False,
        freeze_cls=False,
        cls_forge_weight=0.8,
        patch_forge_weight=0.8,
        cls_loss_weight=1.0,
        # loupe configs - segmentation
        fpn_scales: list[int | float] = [0.5, 2, 4],
        freeze_seg=False,
        tversky_alpha: float = 0.7,
        queries_forge_weight: float = 0.9,
        pixel_forge_weight: float = 0.9,
        pixel_poly_epsilon: float = 1.0,
        seg_loss_weight=1.0,
        mask2former_overrides: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.stage = stage
        if stage not in ["cls", "seg", "cls_seg", "test"]:
            raise ValueError(
                f"stage should be one of ['cls', 'seg', 'cls_seg', 'test'], but got {stage}."
            )

        # basic configs
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.hidden_dropout_prob = hidden_dropout_prob

        # backbone configs
        if all(backbone_name not in name for name in self.supported_backbone):
            raise NotImplementedError(
                f"Backbone {backbone_name} is not supported. "
                f"Please choose from {self.supported_backbone}."
            )

        self.backbone_config: PEConfig = PE_VISION_CONFIG[backbone_name]
        self.backbone_config.output_dim = None  # we use our own linear projection
        for key in backbone_overrides or {}:
            if hasattr(self.backbone_config, key):
                if getattr(backbone_overrides, key) != "-":
                    setattr(self.backbone_config, key, backbone_overrides[key])
            else:
                logger.warning(
                    f"Backbone config {key} is not supported. "
                    f"Please choose from {self.backbone_config.__dict__.keys()}."
                )
        self.backbone_name = backbone_name
        self.backbone_path = backbone_path

        # loupe configs - classification
        self.cls_mlp_ratio = cls_mlp_ratio
        self.cls_mlp_layers = cls_mlp_layers
        self.enable_patch_cls = enable_patch_cls
        self.enable_cls_fusion = enable_cls_fusion
        self.freeze_backbone = freeze_backbone
        self.freeze_cls = freeze_cls
        self.cls_forge_weight = cls_forge_weight
        self.patch_forge_weight = patch_forge_weight
        self.cls_loss_weight = cls_loss_weight

        # loupe configs - segmentation
        self.fpn_scales = sorted(fpn_scales + [1])  # add 1x scale
        if enable_cls_fusion and not enable_patch_cls:
            logger.warning(
                "enable_cls_fusion is set to True, but enable_patch_cls is set to False. "
                "enable_cls_fusion will be ignored."
            )
        self.freeze_seg = freeze_seg
        self.tversky_alpha = tversky_alpha
        self.queries_forge_weight = queries_forge_weight
        self.pixel_forge_weight = pixel_forge_weight
        self.pixel_poly_epsilon = pixel_poly_epsilon
        self.seg_loss_weight = seg_loss_weight

        # remaining configs
        self.hidden_size = self.backbone_config.width
        self.patch_size = self.backbone_config.patch_size
        self.image_size = self.backbone_config.image_size
        self.backbone_output_dim = (
            self.backbone_config.output_dim or self.backbone_config.width
        )
        self.feature_channels = [self.backbone_output_dim] * len(
            self.fpn_scales
        )  # for vit-like backbones, there is only one scale

        # mask2former configs
        overlay = {
            k: may_convert_to_object(v)
            for k, v in (mask2former_overrides or {}).items()
            if v != "-"
        }
        self.mask2former_config = Mask2FormerConfig(**overlay)
        self.label2id = self.mask2former_config.label2id
        self.id2label = self.mask2former_config.id2label

    @property
    def supported_backbone(self):
        supported_backbones = [
            "PE-Core-L16-224",
            "PE-Core-L14-336",
            "PE-Core-G14-448",
            "PE-Spatial-G14-448",
        ]
        return supported_backbones
