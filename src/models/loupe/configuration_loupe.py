import json
import os
from transformers.configuration_utils import PretrainedConfig
from loguru import logger
from models.pe import PE_VISION_CONFIG, PEConfig

class LoupeConfig(PretrainedConfig):
    model_type = "loupe"

    def __init__(
        self,
        backbone_name: str = "PE-Core-L14-336",
        pretrained_path: str = None,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        initializer_range=0.02,
        cls_mlp_hidden_size=2048,  # 2 times of Perception Encoder hidden dim
        cls_mlp_layers=2,
        enable_patch_cls=True,
        enable_cls_fusion=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # basic configs
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.hidden_dropout_prob = hidden_dropout_prob

        # loupe configs
        self.cls_mlp_hidden_size = cls_mlp_hidden_size
        self.cls_mlp_layers = cls_mlp_layers
        self.enable_patch_cls = enable_patch_cls
        self.enable_cls_fusion = enable_cls_fusion
        if enable_cls_fusion and not enable_patch_cls:
            logger.warning(
                "enable_cls_fusion is set to True, but enable_patch_cls is set to False. "
                "enable_cls_fusion will be ignored."
            )

        # backbone configs
        if all(backbone_name not in name for name in self.supported_backbone):
            raise NotImplementedError(
                f"Backbone {backbone_name} is not supported. "
                f"Please choose from {self.supported_backbone}."
            )

        self.backbone_config: PEConfig = PE_VISION_CONFIG[backbone_name]
        self.backbone_config.output_dim = None # we use our own linear projection
        self.pretrained_path = pretrained_path
        self.backbone_name = backbone_name

        # remaining configs
        self.hidden_size = self.backbone_config.width
        self.patch_size = self.backbone_config.patch_size
        self.image_size = self.backbone_config.image_size

    @property
    def supported_backbone(self):
        supported_backbones = [
            "PE-Core-L16-224",
            "PE-Core-L14-336",
            "PE-Core-G14-448",
        ]
        return supported_backbones
