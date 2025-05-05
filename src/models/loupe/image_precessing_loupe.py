from typing import List, Optional, Tuple, Union, Unpack
import PIL
import numpy as np
import torch
from transformers.image_processing_utils_fast import (
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
    BatchFeature,
)
from transformers.image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    PILImageResampling,
)

from models.loupe.modeling_loupe import LoupeUniversalOutput
from .configuration_loupe import LoupeConfig

NonableImageInput = Union[
    "PIL.Image.Image",
    np.ndarray,
    "torch.Tensor",
    list[Optional["PIL.Image.Image"]],
    list[Optional[np.ndarray]],
    list[Optional["torch.Tensor"]],
]


class LoupeImageProcessor(BaseImageProcessorFast):
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"height": 336, "width": 336}
    do_resize = True
    do_rescale = True
    do_normalize = True

    def __init__(
        self, config: LoupeConfig, **kwargs: Unpack[DefaultFastImageProcessorKwargs]
    ):
        super().__init__(**kwargs)
        self.config = config
        self.size = {"height": config.image_size, "width": config.image_size}

    def preprocess_masks(
        self,
        masks: NonableImageInput,
        return_patch_labels: Optional[bool] = None,
        **kwargs: Unpack[DefaultFastImageProcessorKwargs]
    ):
        """
        Preprocess the input masks.
        Args:
            masks (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor` or list of them, item can be None):
                    The input masks to be preprocessed. If a list, it can contain None values.
                    The None values will be replaced with zero tensors of the same shape as the other masks, denoting
                    all pixels are real.
            return_patch_labels (`bool`, *optional*):
                    Whether to return patch labels. If `True`, the output will include patch labels.
                    If `None`, it will use the value of `self.config.enable_patch_cls`.
            kwargs: Additional keyword arguments for preprocessing, but only `device` is used.
        """
        device = kwargs.pop("device", None)
        do_resize = kwargs.pop("do_resize", self.do_resize)
        return_patch_labels = return_patch_labels or self.config.enable_patch_cls

        if isinstance(masks, (list, tuple)):
            none_idx = [i for i, mask in enumerate(masks) if mask is None]
            masks = [mask for mask in masks if mask is not None]
        else:
            none_idx = []

        masks = self.preprocess(
            masks,
            do_normalize=False,
            do_convert_rgb=False,
            device=device,
            do_resize=do_resize,
        )
        pixel_values = masks["pixel_values"]
        for idx in none_idx:
            pixel_values.insert(
                idx,
                torch.zeros(
                    (1, self.size["height"], self.size["width"]), device=device
                ),
            )
        if not return_patch_labels:
            return masks

        patch_size = self.config.patch_size
        patch_labels = []
        for mask in pixel_values:
            C, H, W = mask.shape
            # patch_size * (1, patch_size, W)
            rows = mask.split(patch_size, dim=1)
            patch_label_list = []
            for row in rows:
                # patch_size * (1, patch_size, patch_size)
                patches = row.split(patch_size, dim=2)
                patch_label_list.append(
                    torch.tensor([patch.mean() for patch in patches], device=device)
                )

            # (num_patches, num_patches)
            patch_labels.append(torch.stack(patch_label_list))

        return BatchFeature(
            {"pixel_values": pixel_values, "patch_labels": patch_labels}
        )

    def post_process_segmentation(
        self,
        class_queries_logits: torch.Tensor,
        masks_queries_logits: torch.Tensor,
        target_sizes: Optional[List[Tuple[int, int]]] = None,
    ) -> "torch.Tensor":
        """
        Converts the output of [`LoupeUniversalOutput`] into semantic segmentation maps. Only supports
        PyTorch.

        Args:
            class_queries_logits (`torch.Tensor`, shape `(batch_size, num_queries, num_classes+1)`):
                The class queries logits of the model. The last class is the null class.
            masks_queries_logits (`torch.Tensor`, shape `(batch_size, num_queries, height, width)`):
                The mask queries logits of the model.
            target_sizes (`List[Tuple[int, int]]`, *optional*):
                List of length (batch_size), where each list item (`Tuple[int, int]]`) corresponds to the requested
                final size (height, width) of each prediction. If left to None, predictions will not be resized.
        Returns:
            `List[torch.Tensor]`:
                A list of length `batch_size`, where each item is a semantic segmentation map of shape (height, width)
                corresponding to the target_sizes entry (if `target_sizes` is specified). Each entry of each
                `torch.Tensor` correspond to a semantic class id.
        """
        # Scale back to preprocessed image size - (384, 384) for all models
        masks_queries_logits = torch.nn.functional.interpolate(
            masks_queries_logits, size=(384, 384), mode="bilinear", align_corners=False
        )

        # Remove the null class `[..., :-1]`
        masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
        masks_probs = (
            masks_queries_logits.sigmoid()
        )  # [batch_size, num_queries, height, width]

        # Semantic segmentation logits of shape (batch_size, num_classes, height, width)
        segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
        batch_size = class_queries_logits.shape[0]

        # Resize logits and compute semantic segmentation maps
        if target_sizes is not None:
            if batch_size != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

            semantic_segmentation = []
            for idx in range(batch_size):
                resized_logits = torch.nn.functional.interpolate(
                    segmentation[idx].unsqueeze(dim=0),
                    size=target_sizes[idx],
                    mode="bilinear",
                    align_corners=False,
                )
                semantic_map = resized_logits[0].argmax(dim=0)
                semantic_segmentation.append(semantic_map)
        else:
            semantic_segmentation = segmentation.argmax(dim=1)
            semantic_segmentation = [
                semantic_segmentation[i] for i in range(semantic_segmentation.shape[0])
            ]

        return semantic_segmentation
