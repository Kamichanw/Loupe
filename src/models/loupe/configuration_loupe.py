import json
import os
from transformers.configuration_utils import PretrainedConfig
from loguru import logger


class LoupeConfig(PretrainedConfig):
    model_type = "loupe"

    def __init__(
        self,
        pe_pretrained_path: str = None,
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

        # pe_vision configs
        if not os.path.exists(pe_pretrained_path) or not os.path.exists(
            pe_pretrained_path + "/config.json"
        ) or not os.path.exists(
            pe_pretrained_path + "/model.safetensors"
        ):
            raise FileNotFoundError(
                f"One of pe_pretrained_path {pe_pretrained_path}, config.json, model.safetensors does not exist. "
                "Please make sure you've downloaded checkpoints from https://huggingface.co/facebook/vit_pe_core_large_patch14_336_timm."
            )
        with open(pe_pretrained_path + "/config.json", "r") as f:
            self.pe_vision_config = json.load(f)
        self.pe_pretrained_path = pe_pretrained_path


        # remaining configs
        self.hidden_size = self.pe_vision_config.width
        self.patch_size = self.pe_vision_config.patch_size
        self.image_size = self.pe_vision_config.image_size
