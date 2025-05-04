from typing import Literal, Optional
from transformers.configuration_utils import PretrainedConfig
from transformers.models.mask2former import Mask2FormerConfig
from loguru import logger

from models.pe import PE_VISION_CONFIG, PEConfig


class LoupeConfig(PretrainedConfig):
    model_type = "loupe"

    def __init__(
        self,
        stage: Literal["cls", "seg", "cls_seg", "test"] = "cls",
        # basic configs
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        initializer_range=0.02,
        checkpoint_path: Optional[str] = None,
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
        # loupe configs - segmentation
        fpn_scales: list[int | float] = [0.5, 2, 4],
        freeze_seg=False,
        cls_checkpoint_path: Optional[str] = None,
        mask2former_path: Optional[str] = None,
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
        self.checkpoint_path = checkpoint_path
        if checkpoint_path is not None and (
            backbone_path is not None or mask2former_path is not None
        ):
            logger.warning(
                "checkpoint_path is set, but backbone_path or mask2former_path is also set. "
                "backbone_path and mask2former_path will be ignored."
            )

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
                if getattr(self.backbone_config, key) != "-":
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

        # loupe configs - segmentation
        self.fpn_scales = sorted(fpn_scales + [1])  # add 1x scale
        self.cls_checkpoint_path = cls_checkpoint_path
        if enable_cls_fusion and not enable_patch_cls:
            logger.warning(
                "enable_cls_fusion is set to True, but enable_patch_cls is set to False. "
                "enable_cls_fusion will be ignored."
            )
        self.freeze_seg = freeze_seg
        self.mask2former_path = mask2former_path

        # remaining configs
        self.hidden_size = self.backbone_config.width
        self.patch_size = self.backbone_config.patch_size
        self.image_size = self.backbone_config.image_size
        self.feature_size = (
            self.backbone_config.output_dim or self.backbone_config.width
        )
        self.feature_channels = [self.feature_size] * len(
            self.fpn_scales
        )  # for vit-like backbones, there is only one scale

        # mask2former configs
        overlay = {k: v for k, v in (mask2former_overrides or {}).items() if v != "-"}
        overlay = {
            **dict(
                common_stride=self.patch_size,
                feature_size=self.feature_size,
                feature_strides=[
                    round(scale * self.patch_size) for scale in self.fpn_scales
                ],
            ),
            **overlay,
        }
        self.mask2former_config = Mask2FormerConfig(**overlay)

    @property
    def supported_backbone(self):
        supported_backbones = [
            "PE-Core-L16-224",
            "PE-Core-L14-336",
            "PE-Core-G14-448",
            "PE-Spatial-G14-448",
        ]
        return supported_backbones
