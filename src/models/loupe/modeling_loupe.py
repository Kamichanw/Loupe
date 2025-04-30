from dataclasses import asdict
import os
from loguru import logger
import timm
import torch
import torch.nn as nn

from typing import Optional, Union
from transformers.modeling_utils import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import ImageClassifierOutputWithNoAttention
from transformers.models.mask2former import Mask2FormerForUniversalSegmentation

from models.loupe.loss import LoupeLoss
from models.loupe.configuration_loupe import LoupeConfig
from models.pe import VisionTransformer


class LoupeClassifier(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        hidden_act: Union[str, type] = "gelu",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_dim
        self.num_layers = num_layers

        if num_layers >= 2 and hidden_dim is None:
            raise ValueError(
                "If num_layers >= 2, hidden_dim must be specified. "
                "Otherwise, the model will not be able to learn."
            )

        if isinstance(hidden_act, str):
            self.hidden_act = ACT2FN[hidden_act]
        else:
            self.hidden_act = hidden_act

        self.layers = nn.ModuleList(
            [
                nn.Linear(
                    input_dim if i == 0 else self.hidden_size,
                    self.hidden_size,
                )
                for i in range(self.num_layers - 1)
            ]
        )
        self.layers.append(
            nn.Linear(
                (input_dim if self.num_layers == 1 else self.hidden_size),
                1,  # logits for it is forgery
            )
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            if layer != self.layers[-1]:
                x = self.hidden_act(x)
        return x


class FuseHead(nn.Module):
    def __init__(self, config: LoupeConfig) -> None:
        super().__init__()
        use_cls_token = config.backbone_config.use_cls_token
        num_patches = (config.image_size // config.patch_size) ** 2
        self.fuse = nn.Linear(use_cls_token + num_patches, 1, bias=False)

    def forward(self, x):
        x = self.fuse(x)
        return x


class ScaleBlock(nn.Module):
    """
    Upscale or downscale the input feature map 2x times using nn.ConvTranspose2d or nn.Conv2d.
    """

    def __init__(self, embed_dim, conv1_layer=nn.ConvTranspose2d):
        super().__init__()

        self.conv1 = conv1_layer(
            embed_dim,
            embed_dim,
            kernel_size=2,
            stride=2,
        )
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(
            embed_dim,
            embed_dim,
            kernel_size=3,
            padding=1,
            groups=embed_dim,
            bias=False,
        )
        self.norm = timm.layers.LayerNorm2d(embed_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm(x)

        return x


class LoupeFeaturePyramid(nn.Module):
    def __init__(self, scales: list[int] = None):
        """
        Initializes the LoupeFeaturePyramid with the given scales.

        Args:
            scales (list[int]): A list of integers representing the scales for the pyramid.
                Should be powers of 2 in ascending order. Defaults to [1/2, 1, 2, 4].
        """
        super().__init__()
        self.scales = scales or [1 / 2, 1, 2, 4]

    def forward(self, x):

        return x


class LoupePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LoupeConfig
    base_model_prefix = "loupe"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ResidualAttentionBlock", "Rope2D"]
    _supports_sdpa = True
    _supports_flash_attn_2 = True

    def _init_weights(self, module) -> None:
        """Initialize the weights"""
        if isinstance(module, VisionTransformer):
            return
        elif isinstance(module, FuseHead):
            module.fuse.weight.data = nn.init.constant_(
                module.fuse.weight.data, 1 / module.fuse.in_features
            )
        elif isinstance(module, (nn.Linear)):
            # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
            # `trunc_normal_cpu` not implemented in `half` issues
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()


class LoupeModel(LoupePreTrainedModel):
    def __init__(self, config: LoupeConfig):
        super().__init__(config)
        self.config = config
        self.criterion = LoupeLoss()
        self.backbone = VisionTransformer(**asdict(config.backbone_config))

        backbone_output_dim = (
            config.backbone_config.output_dim or config.backbone_config.width
        )
        self.classifier = LoupeClassifier(
            input_dim=backbone_output_dim,
            hidden_dim=config.cls_mlp_hidden_size,
            num_layers=config.cls_mlp_layers,
            hidden_act=config.hidden_act,
        )
        if config.enable_patch_cls:
            self.patch_classifier = LoupeClassifier(
                input_dim=config.backbone_config.width,
                hidden_dim=config.cls_mlp_hidden_size,
                num_layers=config.cls_mlp_layers,
                hidden_act=config.hidden_act,
            )
            if config.enable_cls_fusion:
                self.fuser = FuseHead(config)

        self.post_init()
        if config.pretrained_path:
            logger.info(f"Loading pretrained weights from {config.pretrained_path}")
            self.backbone.load_ckpt(config.pretrained_path)

    def cls_forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        patch_labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ImageClassifierOutputWithNoAttention]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        patch_labels (`torch.FloatTensor` of shape `(batch_size, num_patches)`):
            Labels for computing the patch-wise classification loss. Each element should be in range of [0, 1], indicating
            the fake pixel ratio of the corresponding patch.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # features: (batch_size, cls_token + num_patches, output_dim)
        features = self.backbone.forward_features(pixel_values, norm=True)
        patch_features = (
            features[:, 1:, :]
            if self.config.backbone_config.use_cls_token
            else features
        )
        output = self.backbone._pool(features)

        # regular classification
        if self.config.backbone_config.pool_type in ["attn", "avg", "tok"]:
            # output: (batch_size, output_dim)
            global_logits = self.classifier(output)
        else:
            if self.config.backbone_config.use_cls_token:
                # output: (batch_size, cls_token + num_patches, output_dim)
                global_logits = self.classifier(output[:, 0, :])
            else:
                raise ValueError("pool_type cannot be none when use_cls_token is False")
        # global_logits: (batch_size, 1)

        # patch classification
        if self.config.enable_patch_cls:
            # patch_logits: (batch_size, num_patches, 1)
            patch_logits = self.patch_classifier(patch_features)
            if self.config.enable_cls_fusion:
                logits = self.fuser(
                    torch.cat([global_logits, patch_logits.squeeze(-1)], dim=1)
                )
            else:
                # regular cls loss will only be on the global logits
                logits = global_logits
        else:
            logits = global_logits
        # logits: (batch_size, 1)

        loss = self.criterion(
            cls_logits=logits,
            cls_labels=labels,
            patch_logits=patch_logits,
            patch_labels=patch_labels,
        )

        if not return_dict:
            return loss, logits

        return ImageClassifierOutputWithNoAttention(
            loss=loss, logits=logits, hidden_states=features
        )

    def seg_forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        patch_labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ImageClassifierOutputWithNoAttention]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        patch_labels (`torch.FloatTensor` of shape `(batch_size, num_patches)`):
            Labels for computing the patch-wise classification loss. Each element should be in range of [0, 1], indicating
            the fake pixel ratio of the corresponding patch.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # features: (batch_size, cls_token + num_patches, output_dim)
        features = self.backbone.forward_features(pixel_values, norm=True)
        patch_features = (
            features[:, 1:, :]
            if self.config.backbone_config.use_cls_token
            else features
        )
        output = self.backbone._pool(features)

        # regular classification
        if self.config.backbone_config.pool_type in ["attn", "avg", "tok"]:
            # output: (batch_size, output_dim)
            global_logits = self.classifier(output)
        else:
            if self.config.backbone_config.use_cls_token:
                # output: (batch_size, cls_token + num_patches, output_dim)
                global_logits = self.classifier(output[:, 0, :])
            else:
                raise ValueError("pool_type cannot be none when use_cls_token is False")
        # global_logits: (batch_size, 1)

        # patch classification
        if self.config.enable_patch_cls:
            # patch_logits: (batch_size, num_patches, 1)
            patch_logits = self.patch_classifier(patch_features)
            if self.config.enable_cls_fusion:
                logits = self.fuser(
                    torch.cat([global_logits, patch_logits.squeeze(-1)], dim=1)
                )
