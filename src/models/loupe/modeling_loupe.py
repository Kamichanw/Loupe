from dataclasses import asdict, dataclass
import math
import os
from loguru import logger
import timm
import torch
import torch.nn as nn

from typing import Dict, List, Optional, Union, cast
from transformers.modeling_utils import PreTrainedModel, ModelOutput
from transformers.activations import ACT2FN
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerPixelDecoder,
    Mask2FormerPixelDecoderOutput,
    Mask2FormerMaskedAttentionDecoderOutput,
    Mask2FormerTransformerModule,
    Mask2FormerPixelDecoderEncoderMultiscaleDeformableAttention,
    Mask2FormerMaskedAttentionDecoderLayer,
    Mask2FormerPixelDecoderEncoderOnly,
)
from transformers.models.mask2former import Mask2FormerConfig


from models.loupe.loss import LoupeClsLoss, LoupeSegLoss
from models.loupe.configuration_loupe import LoupeConfig
from models.pe import VisionTransformer


@dataclass
class LoupeClassificationOutput(ModelOutput):
    """
    Class for Loupe classification outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            The sum of the whole image classification loss and patch classiication (if labels are provided).
        logits (`torch.FloatTensor` of shape `(batch_size, 1)`, *optional*):
            Classification logits of the model, may be fused with patch logits.
        last_hidden_states (`torch.FloatTensor` of shape `(batch_size, num_patches, hidden_dim)`, *optional*):
            Last hidden states of the model (if `output_hidden_states=True`).
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    last_hidden_states: Optional[torch.FloatTensor] = None


@dataclass
class LoupeSegmentationOutput(ModelOutput):
    """
    Class for Loupe segmentation outputs.

    Args:
        loss (`torch.Tensor`, *optional*):
            The computed loss, returned when labels are present.
        class_queries_logits (`torch.FloatTensor`):
            A tensor of shape `(batch_size, num_queries, 1 + 1)` representing the proposed classes for each
            query. Note the `+ 1` is needed because we incorporate the null class.
        masks_queries_logits (`torch.FloatTensor`):
            A tensor of shape `(batch_size, num_queries, height, width)` representing the proposed masks for each
            query.
        auxiliary_logits (`List[Dict(str, torch.FloatTensor)]`, *optional*):
            List of class and mask predictions from each layer of the transformer decoder.
    """

    loss: Optional[torch.FloatTensor] = None
    masks_queries_logits: Optional[torch.FloatTensor] = None
    class_queries_logits: Optional[torch.FloatTensor] = None
    auxiliary_logits: Optional[List[Dict[str, torch.Tensor]]] = None


@dataclass
class LoupeUniversalOutput(ModelOutput):
    """

    Class for Loupe universal outputs.

    Args:
        cls_loss (`torch.FloatTensor`, *optional*):
            The classification loss from LoupeClassificationOutput.
        cls_logits (`torch.FloatTensor`, *optional*):
            Classification logits from LoupeClassificationOutput.
        seg_loss (`torch.FloatTensor`, *optional*):
            The segmentation loss from LoupeSegmentationOutput.
        class_queries_logits (`torch.FloatTensor`, *optional*):
            A tensor of shape `(batch_size, num_queries, 1 + 1)` representing the proposed classes for each
            query. Note the `+ 1` is needed because we incorporate the null class.
        masks_queries_logits (`torch.FloatTensor`, *optional*):
            A tensor of shape `(batch_size, num_queries, height, width)` representing the proposed masks for each
            query.
    """

    cls_loss: Optional[torch.FloatTensor] = None
    cls_logits: Optional[torch.FloatTensor] = None
    seg_loss: Optional[torch.FloatTensor] = None
    class_queries_logits: Optional[torch.FloatTensor] = None
    masks_queries_logits: Optional[torch.FloatTensor] = None


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

    def init_tensors(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            if layer != self.layers[-1]:
                x = self.hidden_act(x)
        return x


class FuseHead(nn.Module):
    def __init__(self, config: LoupeConfig) -> None:
        super().__init__()
        num_patches = (config.image_size // config.patch_size) ** 2
        self.fuse = nn.Linear(1 + num_patches, 1, bias=False)

    def init_tensors(self):
        self.fuse.weight.data = nn.init.constant_(
            self.fuse.weight.data, 1 / self.fuse.in_features
        )

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
        if conv1_layer == nn.ConvTranspose2d:
            channel_args = (
                n_channels,
                n_channels,
            )
        else:
            channel_args = ()
        self.conv1 = conv1_layer(
            *channel_args,
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
            [
                nn.Sequential(*self._make_layer(scale))
                for scale in sorted(self.scales, reverse=True)
            ]
        )

    def _make_layer(self, scale: float | int) -> list[nn.Module]:
        if scale == 1:
            return [nn.Identity()]
        conv1_layer = nn.ConvTranspose2d if scale > 1 else nn.MaxPool2d
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
        mask2former_config = config.mask2former_config
        self.pixel_decoder = Mask2FormerPixelDecoder(
            mask2former_config, config.feature_channels
        )
        self.mask2former_decoder = Mask2FormerTransformerModule(
            in_features=mask2former_config.feature_size, config=mask2former_config
        )

        # this is a bug of transformers library
        # the input_projections should be a nn.ModuleList instead of a list
        if isinstance(self.mask2former_decoder.input_projections, list):
            self.mask2former_decoder.input_projections = nn.ModuleList(
                self.mask2former_decoder.input_projections
            )

        self.class_predictor = nn.Linear(
            mask2former_config.hidden_dim, mask2former_config.num_labels + 1
        )

        self.criterion = LoupeSegLoss(config=config)

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
        for key, weight in self.criterion.weight_dict.items():
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
        features: torch.Tensor,
        mask_labels: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None,
    ):
        # scale features to the different scales
        backbone_features = self.fpn(features)
        pixel_decoder_output: Mask2FormerPixelDecoderOutput = self.pixel_decoder(
            backbone_features, output_hidden_states=False
        )

        transformer_module_output: Mask2FormerMaskedAttentionDecoderOutput = (
            self.mask2former_decoder(
                multi_scale_features=pixel_decoder_output.multi_scale_features,
                mask_features=pixel_decoder_output.mask_features,
                output_hidden_states=self.config.mask2former_config.use_auxiliary_loss,
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

        return LoupeSegmentationOutput(
            loss=loss,
            class_queries_logits=class_queries_logits[-1],
            masks_queries_logits=masks_queries_logits[-1],
            auxiliary_logits=auxiliary_logits,
        )


class LoupePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LoupeConfig
    base_model_prefix = "loupe"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _supports_flash_attn_2 = True

    def _init_weights(self, module) -> None:
        """Initialize the weights"""
        config = cast(LoupeConfig, self.config)
        xavier_std = config.mask2former_config.init_xavier_std
        std = config.initializer_range
        if isinstance(module, (VisionTransformer, FuseHead, LoupeClassifier)):
            module.init_tensors()
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
        if config.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        if "test" in config.stage or "cls" in config.stage:
            backbone_output_dim = config.backbone_output_dim
            self.classifier = LoupeClassifier(
                input_dim=backbone_output_dim,
                hidden_dim=backbone_output_dim * config.cls_mlp_ratio,
                num_layers=config.cls_mlp_layers,
                hidden_act=config.hidden_act,
            )
            if config.enable_patch_cls:
                self.patch_classifier = LoupeClassifier(
                    input_dim=config.backbone_config.width,
                    hidden_dim=config.backbone_config.width * config.cls_mlp_ratio,
                    num_layers=config.cls_mlp_layers,
                    hidden_act=config.hidden_act,
                )
                if config.enable_cls_fusion:
                    self.fuser = FuseHead(config)
            self.cls_criterion = LoupeClsLoss()

            if config.freeze_cls:
                for param in self.classifier.parameters():
                    param.requires_grad = False
                if config.enable_patch_cls:
                    for param in self.patch_classifier.parameters():
                        param.requires_grad = False
                    if config.enable_cls_fusion:
                        for param in self.fuser.parameters():
                            param.requires_grad = False

        if "seg" in config.stage or "test" in config.stage:
            self.segmentor = LoupeSegmentor(config)

            if config.freeze_seg:
                for param in self.segmentor.parameters():
                    param.requires_grad = False

        self.post_init()

        # load checkpoints
        if config.backbone_path:
            logger.info(f"Loading backbone from {config.backbone_path}")
            if config.backbone_path.endswith(".pt"):
                self.backbone.load_ckpt(config.backbone_path)
            elif config.backbone_path.endswith(".safetensors"):
                from safetensors.torch import load_file

                state_dict = load_file(config.backbone_path)
                self.load_state_dict(
                    {k.removeprefix("loupe."): v for k, v in state_dict.items()},
                    strict=False,
                )

    def cls_forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        patch_labels: Optional[torch.Tensor] = None,
    ) -> LoupeClassificationOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        patch_labels (`torch.FloatTensor` of shape `(batch_size, num_patches)`):
            Labels for computing the patch-wise classification loss. Each element should be in range of [0, 1], indicating
            the fake pixel ratio of the corresponding patch.
        labels (`torch.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for each image in the batch, indicating whether the image is forged.
        """

        # features: (batch_size, cls_token + num_patches, output_dim)
        features = self.backbone.forward_features(pixel_values, norm=True)

        loss, logits, patch_logits = None, None, None
        if labels is not None:
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
                    raise ValueError(
                        "pool_type cannot be none when use_cls_token is False"
                    )
            # global_logits: (batch_size, 1)

            # patch classification
            if self.config.enable_patch_cls:
                patch_features = (
                    features[:, 1:, :]
                    if self.config.backbone_config.use_cls_token
                    else features
                )
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

            loss = self.cls_criterion(
                cls_logits=logits,
                cls_labels=labels,
                patch_logits=patch_logits,
                patch_labels=patch_labels,
            )

        return LoupeClassificationOutput(
            loss=loss,
            logits=logits,
            last_hidden_states=features,
        )

    def seg_forward(
        self,
        features: List[torch.Tensor],
        masks: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> LoupeSegmentationOutput:
        r"""
        features (`torch.Tensor` of shape `(batch_size, num_patches, hidden_dim)`, *optional*):
            Features of the input image extracted by backbone.
        masks (`torch.Tensor` of shape `(batch_size, height, width)`, *optional*):
            Segmentation masks for each image in the batch. Each mask should be in range of [0, 1], indicating the
            forgery ratio of a pixel.
        labels (`torch.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for each image in the batch, indicating whether the image is forged.
        """
        # (batch_size, height, width) -> (batch_size, num_labels=1, height, width)
        masks = masks.unsqueeze(1) if masks is not None else None
        # (batch_size,) -> (batch_size, num_labels=1)
        labels = labels.unsqueeze(1) if labels is not None else None

        return self.segmentor(
            features=features,
            mask_labels=masks,
            class_labels=labels,
        )

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        patch_labels: Optional[torch.Tensor] = None,
    ) -> LoupeUniversalOutput:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(batch_size, 3, height, width)`):
            Pixel values of the input image. Should be of the same size as the input image.
        masks (`torch.FloatTensor` of shape `(batch_size, height, width)`, *optional*):
            Segmentation masks for each image in the batch. Each mask should be in range of [0, 1], indicating the
            forgery ratio of a pixel.
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for each image in the batch, indicating whether the image is forged.
        patch_labels (`torch.FloatTensor` of shape `(batch_size, num_patches)`, *optional*):
            Labels for computing the patch-wise classification loss. Each element should be in range of [0, 1], indicating
            the forgery ratio of the corresponding patch.
        """
        cls_loss, cls_logits, seg_loss, class_queries_logits, masks_queries_logits = (
            None,
            None,
            None,
            None,
            None,
        )

        def reshape_features(features: torch.Tensor) -> torch.Tensor:
            if self.config.backbone_config.use_cls_token:
                # (batch_size, cls_token + num_patches, hidden_dim) -> (batch_size, num_patches, hidden_dim)
                features = features[:, 1:, :]
            height = width = (
                self.config.backbone_config.image_size // self.config.patch_size
            )
            # (batch_size, num_patches, hidden_dim) -> (batch_size, hidden_dim, height, width)
            return features.view(
                features.shape[0],
                height,
                width,
                features.shape[-1],
            ).permute(0, 3, 1, 2)

        if self.config.stage == "cls":
            output = self.cls_forward(
                pixel_values=pixel_values,
                labels=labels,
                patch_labels=patch_labels,
            )
            cls_loss = output.loss
            cls_logits = output.logits
        elif self.config.stage == "seg":
            cls_output = self.cls_forward(pixel_values)
            output = self.seg_forward(
                features=reshape_features(cls_output.last_hidden_states),
                masks=masks,
                labels=labels,
            )
            seg_loss = output.loss
            masks_queries_logits = output.masks_queries_logits
            class_queries_logits = output.class_queries_logits
        else:
            cls_output = self.cls_forward(pixel_values, labels, patch_labels)
            seg_output = self.seg_forward(
                features=reshape_features(cls_output.last_hidden_states),
                masks=masks,
                labels=labels,
            )
            cls_loss = cls_output.loss
            cls_logits = cls_output.logits
            seg_loss = seg_output.loss
            masks_queries_logits = seg_output.masks_queries_logits
            class_queries_logits = seg_output.class_queries_logits

        return LoupeUniversalOutput(
            cls_loss=cls_loss,
            cls_logits=cls_logits,
            seg_loss=seg_loss,
            masks_queries_logits=masks_queries_logits,
            class_queries_logits=class_queries_logits,
        )
