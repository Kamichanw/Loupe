from dataclasses import asdict
import math
import os
from loguru import logger
import timm
import torch
import torch.nn as nn

from typing import Dict, List, Optional, Union
from transformers.modeling_utils import PreTrainedModel, ModelOutput
from transformers.activations import ACT2FN
from transformers.modeling_outputs import ImageClassifierOutputWithNoAttention
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerPixelDecoder,
    Mask2FormerPixelDecoderOutput,
    Mask2FormerModelOutput,
    Mask2FormerMaskedAttentionDecoderOutput,
    Mask2FormerPixelLevelModuleOutput,
    Mask2FormerTransformerModule,
    Mask2FormerPixelDecoderEncoderMultiscaleDeformableAttention,
    Mask2FormerMaskedAttentionDecoderLayer,
    Mask2FormerPixelDecoderEncoderOnly,
)
from transformers.models.mask2former import Mask2FormerConfig


from models.loupe.loss import LoupeClsLoss
from models.loupe.configuration_loupe import LoupeConfig
from models.pe import VisionTransformer


class LoupeClassificationOutput(ModelOutput):
    """
    Class for Loupe classification outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            The sum of the whole image classification loss and patch classiication (if labels are provided).
        cls_logits (`torch.FloatTensor` of shape `(batch_size, 1)`, *optional*):
            Classification logits (if labels are provided).
        patch_logits (`torch.FloatTensor` of shape `(batch_size, num_patches, 1)`, *optional*):
            Patch classification logits (if labels are provided).
        last_hidden_states (`torch.FloatTensor` of shape `(batch_size, num_patches, hidden_dim)`, *optional*):
            Last hidden states of the model (if `output_hidden_states=True`).
    """

    loss: Optional[torch.FloatTensor] = None
    cls_logits: Optional[torch.FloatTensor] = None
    patch_logits: Optional[torch.FloatTensor] = None
    last_hidden_states: Optional[torch.FloatTensor] = None


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
                1,  # logits for predicting if it is forgery
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

    def __init__(
        self,
        n_channels,
        conv1_layer=nn.ConvTranspose2d,
        hidden_act: Union[str, type] = "gelu",
    ) -> None:
        super().__init__()

        self.conv1 = conv1_layer(
            n_channels,
            n_channels,
            kernel_size=2,
            stride=2,
        )
        if isinstance(hidden_act, str):
            self.hidden_act = ACT2FN[hidden_act]
        else:
            self.hidden_act = hidden_act
        self.conv2 = nn.Conv2d(
            n_channels,
            n_channels,
            kernel_size=3,
            padding=1,
            groups=n_channels,
            bias=False,
        )
        self.norm = timm.layers.LayerNorm2d(n_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.hidden_act(x)
        x = self.conv2(x)
        x = self.norm(x)

        return x


class LoupeFeaturePyramid(nn.Module):

    def __init__(self, n_channels: int, scales: list[float | int] = None):
        """
        Initializes the LoupeFeaturePyramid with the given scales.

        Args:
            n_channels (int): The number of channels in the input feature map.
            scales (list[float or int]): A list whose length=4 representing the scales for the pyramid.
                Should be integer powers of 2 in ascending order. Defaults to [1/2, 1, 2, 4].
        """
        super().__init__()
        self.hidden_dim = n_channels
        self.scales = scales or [1 / 2, 1, 2, 4]
        is_power_of_2 = lambda n: n > 0 and math.isclose(
            math.log2(n), round(math.log2(n))
        )
        if any(not is_power_of_2(scale) for scale in self.scales):
            raise ValueError(
                f"All scales must be integer powers of 2, but got {self.scales}"
            )

        self.scale_layers = nn.ModuleList(
            [nn.Sequential(*self._make_layer(scale)) for scale in self.scales]
        )

    def _make_layer(self, scale: float | int) -> list[nn.Module]:
        if scale == 1:
            return [nn.Identity()]
        conv1_layer = nn.ConvTranspose2d if scale > 1 else nn.Conv2d
        num_steps = abs(int(round(math.log2(scale))))
        return [
            ScaleBlock(self.hidden_dim, conv1_layer=conv1_layer)
            for _ in range(num_steps)
        ]

    def forward(self, x):
        return [layer(x) for layer in self.scale_layers]


class LoupeSegmentor(nn.Module):
    def __init__(self, config: LoupeConfig):
        super().__init__()
        self.config = config
        self.fpn = LoupeFeaturePyramid(config.backbone_config.width)
        self.pixel_decoder = Mask2FormerPixelDecoder(
            config.mask2former_config, config.feature_channels
        )
        self.mask2former_decoder = Mask2FormerTransformerModule(
            in_features=config.feature_size, config=config.mask2former_config
        )
        self.weight_dict: Dict[str, float] = {
            "loss_cross_entropy": config.class_weight,
            "loss_mask": config.mask_weight,
            "loss_dice": config.dice_weight,
        }

        self.class_predictor = nn.Linear(config.hidden_dim, config.num_labels + 1)

        self.criterion = Mask2FormerLoss(config=config, weight_dict=self.weight_dict)

    def get_loss_dict(
        self,
        masks_queries_logits: torch.Tensor,
        class_queries_logits: torch.Tensor,
        mask_labels: torch.Tensor,
        class_labels: torch.Tensor,
        auxiliary_predictions: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        loss_dict: Dict[str, torch.Tensor] = self.criterion(
            masks_queries_logits=masks_queries_logits,
            class_queries_logits=class_queries_logits,
            mask_labels=mask_labels,
            class_labels=class_labels,
            auxiliary_predictions=auxiliary_predictions,
        )

        # weight each loss by `self.weight_dict[<LOSS_NAME>]` including auxiliary losses
        for key, weight in self.weight_dict.items():
            for loss_key, loss in loss_dict.items():
                if key in loss_key:
                    loss *= weight

        return loss_dict

    def get_loss(self, loss_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        return sum(loss_dict.values())

    def get_auxiliary_logits(self, classes: torch.Tensor, output_masks: torch.Tensor):
        auxiliary_logits: List[Dict[str, torch.Tensor]] = []

        for aux_binary_masks, aux_classes in zip(output_masks[:-1], classes[:-1]):
            auxiliary_logits.append(
                {
                    "masks_queries_logits": aux_binary_masks,
                    "class_queries_logits": aux_classes,
                }
            )

        return auxiliary_logits

    def forward(
        self,
        backbone_features: List[torch.Tensor],
        return_dict: Optional[bool] = None,
    ):
        pixel_decoder_output: Mask2FormerPixelDecoderOutput = self.pixel_decoder(
            backbone_features, output_hidden_states=False
        )

        transformer_module_output: Mask2FormerMaskedAttentionDecoderOutput = (
            self.mask2former_decoder(
                multi_scale_features=pixel_decoder_output.multi_scale_features,
                mask_features=pixel_decoder_output.mask_features,
                output_hidden_states=True,
                output_attentions=False,
            )
        )

        loss, loss_dict, auxiliary_logits = None, None, None
        class_queries_logits = ()

        for decoder_output in transformer_module_output.intermediate_hidden_states:
            class_prediction = self.class_predictor(decoder_output.transpose(0, 1))
            class_queries_logits += (class_prediction,)

        masks_queries_logits = transformer_module_output.masks_queries_logits

        auxiliary_logits = self.get_auxiliary_logits(
            class_queries_logits, masks_queries_logits
        )

        if mask_labels is not None and class_labels is not None:
            loss_dict = self.get_loss_dict(
                masks_queries_logits=masks_queries_logits[-1],
                class_queries_logits=class_queries_logits[-1],
                mask_labels=mask_labels,
                class_labels=class_labels,
                auxiliary_predictions=auxiliary_logits,
            )
            loss = self.get_loss(loss_dict)

        encoder_hidden_states = None
        pixel_decoder_hidden_states = None
        transformer_decoder_hidden_states = None

        output_auxiliary_logits = (
            self.config.output_auxiliary_logits
            if output_auxiliary_logits is None
            else output_auxiliary_logits
        )
        if not output_auxiliary_logits:
            auxiliary_logits = None

        output = Mask2FormerForUniversalSegmentationOutput(
            loss=loss,
            class_queries_logits=class_queries_logits[-1],
            masks_queries_logits=masks_queries_logits[-1],
            auxiliary_logits=auxiliary_logits,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            pixel_decoder_last_hidden_state=outputs.pixel_decoder_last_hidden_state,
            transformer_decoder_last_hidden_state=outputs.transformer_decoder_last_hidden_state,
            encoder_hidden_states=encoder_hidden_states,
            pixel_decoder_hidden_states=pixel_decoder_hidden_states,
            transformer_decoder_hidden_states=transformer_decoder_hidden_states,
            attentions=outputs.attentions,
        )

        if not return_dict:
            output = tuple(v for v in output.values() if v is not None)
            if loss is not None:
                output = (loss) + output
        return output

        if not return_dict:
            output = tuple(v for v in output.values() if v is not None)

        return output


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
        xavier_std = 1.0
        std = self.config.initializer_range
        if isinstance(module, (VisionTransformer, FuseHead, LoupeClassifier)):
            return
        elif isinstance(module, Mask2FormerTransformerModule):
            if module.input_projections is not None:
                for input_projection in module.input_projections:
                    if not isinstance(input_projection, nn.Sequential):
                        nn.init.xavier_uniform_(
                            input_projection.weight, gain=xavier_std
                        )
                        nn.init.constant_(input_projection.bias, 0)

        elif isinstance(
            module, Mask2FormerPixelDecoderEncoderMultiscaleDeformableAttention
        ):
            nn.init.constant_(module.sampling_offsets.weight.data, 0.0)
            thetas = torch.arange(module.n_heads, dtype=torch.int64).float() * (
                2.0 * math.pi / module.n_heads
            )
            grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
            grid_init = (
                (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
                .view(module.n_heads, 1, 1, 2)
                .repeat(1, module.n_levels, module.n_points, 1)
            )
            for i in range(module.n_points):
                grid_init[:, :, i, :] *= i + 1
            with torch.no_grad():
                module.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))

            nn.init.constant_(module.attention_weights.weight.data, 0.0)
            nn.init.constant_(module.attention_weights.bias.data, 0.0)
            nn.init.xavier_uniform_(module.value_proj.weight.data)
            nn.init.constant_(module.value_proj.bias.data, 0.0)
            nn.init.xavier_uniform_(module.output_proj.weight.data)
            nn.init.constant_(module.output_proj.bias.data, 0.0)

        elif isinstance(module, Mask2FormerMaskedAttentionDecoderLayer):
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p, gain=xavier_std)
        elif isinstance(module, Mask2FormerPixelDecoder):
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            nn.init.normal_(module.level_embed, std=0)

        elif isinstance(module, Mask2FormerPixelDecoderEncoderOnly):
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        elif isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        if hasattr(module, "reference_points"):
            nn.init.xavier_uniform_(module.reference_points.weight.data, gain=1.0)
            nn.init.constant_(module.reference_points.bias.data, 0.0)


class LoupeModel(LoupePreTrainedModel):
    def __init__(self, config: LoupeConfig):
        super().__init__(config)
        self.config = config

        # init backbone
        self.backbone = VisionTransformer(**asdict(config.backbone_config))
        if config.backbone_path:
            logger.info(f"Loading pretrained weights from {config.backbone_path}")
            self.backbone.load_ckpt(config.backbone_path)
        if config.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        

        if config.stage in ["cls", "test"]:
            backbone_output_dim = config.feature_size
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
            self.cls_criterion = LoupeClsLoss()
            # TODO: add load checkpoint

            if config.freeze_cls:
                for param in self.classifier.parameters():
                    param.requires_grad = False
                if config.enable_patch_cls:
                    for param in self.patch_classifier.parameters():
                        param.requires_grad = False
                if config.enable_cls_fusion:
                    for param in self.fuser.parameters():
                        param.requires_grad = False

        if config.stage in ["seg", "test"]:
            self.segmentor = LoupeSegmentor(config)
            # TODO: add load checkpoint

            if config.freeze_seg:
                for param in self.segmentor.parameters():
                    param.requires_grad = False

        self.post_init()

    def cls_forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        patch_labels: Optional[torch.Tensor] = None,
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

        loss = (
            self.cls_criterion(
                cls_logits=logits,
                cls_labels=labels,
                patch_logits=patch_logits,
                patch_labels=patch_labels,
            )
            if labels is not None
            else None
        )

        return LoupeClassificationOutput(
            loss=loss,
            cls_logits=global_logits,
            patch_logits=patch_logits,
            last_hidden_states=features,
        )

    def seg_forward(
        self,
        features: List[torch.Tensor],
        masks: torch.Tensor,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ImageClassifierOutputWithNoAttention]:
        r"""
        features (`torch.Tensor` of shape `(batch_size, num_patches, hidden_dim)`, *optional*):
            Features of the input image extracted by backbone.
        masks (`torch.Tensor` of shape `(batch_size, num_patches, num_patches)`, *optional*):
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
