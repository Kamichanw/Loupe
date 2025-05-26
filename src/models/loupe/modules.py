import math
import torch
import torch.nn as nn
import timm

from typing import Optional, Union
from transformers.activations import ACT2FN
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerPixelDecoder,
    Mask2FormerPixelDecoderOutput,
    Mask2FormerPixelDecoderEncoderOnly,
    Mask2FormerPixelDecoderEncoderLayer,
    Mask2FormerPixelDecoderEncoderMultiscaleDeformableAttention,
)
from transformers.modeling_outputs import BaseModelOutput

from models.loupe.configuration_loupe import LoupeConfig


# ----------------------------------- CLASSIFICATION-RELATED MODULES -----------------------------------
class LoupeClsHead(nn.Module):

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
        nn.init.constant_(self.fuse.weight.data, 1 / self.fuse.in_features)

    def forward(self, x):
        x = self.fuse(x)
        return x


# ----------------------------------- SEGMENTATION-RELATED MODULES -----------------------------------
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
        x = self.conv2(
            x.contiguous()
        )  # who knows why I have to add contiguous here ?????
        x = self.norm(x)

        return x


class FeaturePyramid(nn.Module):

    def __init__(self, n_channels: int, scales: list[float | int] = None):
        """
        Initializes the FeaturePyramid with the given scales.

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


class PixelDecoderEncoderLayer(Mask2FormerPixelDecoderEncoderLayer):
    def __init__(self, config: LoupeConfig):
        mask2former_config = config.mask2former_config
        super().__init__(mask2former_config)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=mask2former_config.num_attention_heads,
            batch_first=True,
        )
        self.cross_attn_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        conditional_queries: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        reference_points=None,
        spatial_shapes_list=None,
        level_start_index=None,
        output_attentions: bool = False,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Input to the layer.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                Attention mask.
            conditional_queries (`torch.FloatTensor` of shape `(batch_size, 1, hidden_size)`):
                Pseudo query for the cross attention, added from Loupe.
            position_embeddings (`torch.FloatTensor`, *optional*):
                Position embeddings, to be added to `hidden_states`.
            reference_points (`torch.FloatTensor`, *optional*):
                Reference points.
            spatial_shapes_list (`list` of `tuple`):
                Spatial shapes of the backbone feature maps as a list of tuples.
            level_start_index (`torch.LongTensor`, *optional*):
                Level start index.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        # Apply Multi-scale Deformable Attention Module on the multi-scale feature maps.
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            reference_points=reference_points,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
            output_attentions=output_attentions,
        )

        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        if conditional_queries is not None:
            residual = hidden_states
            # Apply cross attention on the pseudo query.
            hidden_states, _ = self.cross_attn(
                query=conditional_queries,
                key=hidden_states,
                value=hidden_states,
                key_padding_mask=attention_mask,
                need_weights=False,
            )
            hidden_states = nn.functional.dropout(
                hidden_states, p=self.dropout, training=self.training
            )
            hidden_states = residual + hidden_states
            hidden_states = self.cross_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.activation_dropout, training=self.training
        )

        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )

        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if self.training:
            if torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(
                    hidden_states, min=-clamp_value, max=clamp_value
                )

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights.transpose(1, 0),)

        return outputs


class PixelDecoderConditionalEncoder(Mask2FormerPixelDecoderEncoderOnly):
    def __init__(self, config: LoupeConfig):
        mask2former_config = config.mask2former_config
        super().__init__(mask2former_config)
        # replace the original encoder layer with our loupe encoder layer
        self.layers = nn.ModuleList(
            [
                PixelDecoderEncoderLayer(config)
                for _ in range(mask2former_config.encoder_layers)
            ]
        )

    def forward(
        self,
        inputs_embeds=None,
        attention_mask=None,
        conditional_queries=None,
        position_embeddings=None,
        spatial_shapes_list=None,
        level_start_index=None,
        valid_ratios=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Flattened feature map (output of the backbone + projection layer) that is passed to the encoder.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding pixel features. Mask values selected in `[0, 1]`:
                - 1 for pixel features that are real (i.e. **not masked**),
                - 0 for pixel features that are padding (i.e. **masked**).
                [What are attention masks?](../glossary#attention-mask)
            conditional_queries (`torch.FloatTensor` of shape `(batch_size, 1, hidden_size)`):
                Conditional query for the cross attention, added from Loupe.
            position_embeddings (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Position embeddings that are added to the queries and keys in each self-attention layer.
            spatial_shapes_list (`list` of `tuple`):
                Spatial shapes of each feature map as a list of tuples.
            level_start_index (`torch.LongTensor` of shape `(num_feature_levels)`):
                Starting index of each feature map.
            valid_ratios (`torch.FloatTensor` of shape `(batch_size, num_feature_levels, 2)`):
                Ratio of valid area in each feature level.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        hidden_states = inputs_embeds
        reference_points = self.get_reference_points(
            spatial_shapes_list, valid_ratios, device=inputs_embeds.device
        )

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for i, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states.transpose(1, 0),)

            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask,
                conditional_queries=conditional_queries,
                position_embeddings=position_embeddings,
                reference_points=reference_points,
                spatial_shapes_list=spatial_shapes_list,
                level_start_index=level_start_index,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states += (hidden_states.transpose(1, 0),)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class PixelDecoder(Mask2FormerPixelDecoder):
    def __init__(self, config: LoupeConfig):
        mask2former_config = config.mask2former_config
        super().__init__(mask2former_config, config.feature_channels)

        # replace the original encoder with our loupe conditional encoder
        self.encoder = PixelDecoderConditionalEncoder(config)

    # modified from transformers.models.mask2former.modeling_mask2former.Mask2FormerPixelDecoder
    def forward(
        self,
        features,
        conditional_queries=None,
        encoder_outputs=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        # Apply 1x1 convolution to reduce the channel dimension to d_model (256 by default)
        input_embeds = []
        position_embeddings = []
        for level, x in enumerate(features[::-1][: self.num_feature_levels]):
            input_embeds.append(self.input_projections[level](x))
            position_embeddings.append(self.position_embedding(x))

        masks = [
            torch.zeros(
                (x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool
            )
            for x in input_embeds
        ]

        # Prepare encoder inputs (by flattening)
        spatial_shapes_list = [
            (embed.shape[2], embed.shape[3]) for embed in input_embeds
        ]
        input_embeds_flat = torch.cat(
            [embed.flatten(2).transpose(1, 2) for embed in input_embeds], 1
        )
        spatial_shapes = torch.as_tensor(
            spatial_shapes_list, dtype=torch.long, device=input_embeds_flat.device
        )
        masks_flat = torch.cat([mask.flatten(1) for mask in masks], 1)

        position_embeddings = [
            embed.flatten(2).transpose(1, 2) for embed in position_embeddings
        ]
        level_pos_embed_flat = [
            x + self.level_embed[i].view(1, 1, -1)
            for i, x in enumerate(position_embeddings)
        ]
        level_pos_embed_flat = torch.cat(level_pos_embed_flat, 1)

        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        valid_ratios = torch.stack(
            [
                self.get_valid_ratio(mask, dtype=input_embeds_flat.dtype)
                for mask in masks
            ],
            1,
        )

        # Send input_embeds_flat + masks_flat + level_pos_embed_flat (backbone + proj layer output) through encoder
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                inputs_embeds=input_embeds_flat,
                attention_mask=masks_flat,
                conditional_queries=conditional_queries,
                position_embeddings=level_pos_embed_flat,
                spatial_shapes_list=spatial_shapes_list,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        last_hidden_state = encoder_outputs.last_hidden_state
        batch_size = last_hidden_state.shape[0]

        # We compute level_start_index_list separately from the tensor version level_start_index
        # to avoid iterating over a tensor which breaks torch.compile/export.
        level_start_index_list = [0]
        for height, width in spatial_shapes_list[:-1]:
            level_start_index_list.append(level_start_index_list[-1] + height * width)
        split_sizes = [None] * self.num_feature_levels
        for i in range(self.num_feature_levels):
            if i < self.num_feature_levels - 1:
                split_sizes[i] = (
                    level_start_index_list[i + 1] - level_start_index_list[i]
                )
            else:
                split_sizes[i] = last_hidden_state.shape[1] - level_start_index_list[i]

        encoder_output = torch.split(last_hidden_state, split_sizes, dim=1)

        # Compute final features
        outputs = [
            x.transpose(1, 2).view(
                batch_size, -1, spatial_shapes_list[i][0], spatial_shapes_list[i][1]
            )
            for i, x in enumerate(encoder_output)
        ]

        # Append extra FPN levels to outputs, ordered from low to high resolution
        for idx, feature in enumerate(features[: self.num_fpn_levels][::-1]):
            lateral_conv = self.lateral_convolutions[idx]
            output_conv = self.output_convolutions[idx]
            current_fpn = lateral_conv(feature)

            # Following FPN implementation, we use nearest upsampling here
            out = current_fpn + nn.functional.interpolate(
                outputs[-1],
                size=current_fpn.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            out = output_conv(out)
            outputs.append(out)

        num_cur_levels = 0
        multi_scale_features = []

        for out in outputs:
            if num_cur_levels < self.num_feature_levels:
                multi_scale_features.append(out)
                num_cur_levels += 1

        return Mask2FormerPixelDecoderOutput(
            mask_features=self.mask_projection(outputs[-1]),
            multi_scale_features=tuple(multi_scale_features),
            attentions=encoder_outputs.attentions,
        )
