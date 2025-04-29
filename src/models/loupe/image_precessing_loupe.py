from typing import Optional, Union, Unpack
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
        **kwargs: Unpack[DefaultFastImageProcessorKwargs]
    ):
        """
        Preprocess the input masks.
        Args:
            masks (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor` or list of them, item can be None):
                    The input masks to be preprocessed. If a list, it can contain None values.
                    The None values will be replaced with zero tensors of the same shape as the other masks, denoting
                    all pixels are real.
            kwargs: Additional keyword arguments for preprocessing, but only `device` is used.
        """
        device = kwargs.pop("device", None)
        if isinstance(masks, (list, tuple)):
            none_idx = [i for i, mask in enumerate(masks) if mask is None]
            masks = [mask for mask in masks if mask is not None]
        else:
            none_idx = []

        masks = self.preprocess(
            masks, do_normalize=False, do_convert_rgb=False, device=device
        )
        pixel_values = masks["pixel_values"]
        for idx in none_idx:
            pixel_values.insert(
                idx,
                torch.zeros(
                    (1, self.size["height"], self.size["width"]), device=device
                ),
            )
        if not self.config.enable_patch_cls:
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
