import random
from typing import Dict, List, Optional, Tuple, Union, Unpack
import PIL
import PIL.Image
import numpy as np
import torch
import torch.nn.functional as F
from transformers.models.mask2former import Mask2FormerImageProcessor
from transformers.image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    PILImageResampling,
    ImageInput,
)

from .modeling_loupe import LoupeUniversalOutput
from .configuration_loupe import LoupeConfig

NonableImageInput = Union[
    "PIL.Image.Image",
    np.ndarray,
    "torch.Tensor",
    list[Optional["PIL.Image.Image"]],
    list[Optional[np.ndarray]],
    list[Optional["torch.Tensor"]],
]


class LoupeImageProcessor(Mask2FormerImageProcessor):
    def __init__(
        self,
        config: LoupeConfig,
        size: Dict[str, int] = None,
        do_resize=True,
        do_rescale=True,
        do_normalize=True,
        do_reduce_labels=True,
        image_mean: Union[float, List[float]] = IMAGENET_STANDARD_MEAN,
        image_std: Union[float, List[float]] = IMAGENET_STANDARD_STD,
        ignore_index: int = 255,
        size_divisor: int = 0,
        **kwargs
    ):
        super().__init__(
            size=size or {"height": config.image_size, "width": config.image_size},
            do_resize=do_resize,
            do_rescale=do_rescale,
            do_normalize=do_normalize,
            do_reduce_labels=do_reduce_labels,
            image_mean=image_mean,
            image_std=image_std,
            ignore_index=ignore_index,
            size_divisor=size_divisor,
            **kwargs
        )
        self.config = config

    def convert_to_binary_masks(
        self, masks: ImageInput, return_normalized_mask: bool = False
    ):
        """convert masks to binary mask with 0 and 1"""
        is_batched = isinstance(masks, (list, tuple))
        if not is_batched:
            masks = [masks]
        normalized_mask = []
        for i, mask in enumerate(masks):
            if isinstance(mask, PIL.Image.Image):
                mask = np.array(mask.convert("L"))
            if isinstance(mask, np.ndarray):
                mask = torch.tensor(mask)
            min_val, max_val = mask.min(), mask.max()
            if max_val - min_val > 0:
                mask = (mask - min_val) / (max_val - min_val)
            else:
                mask = torch.zeros_like(mask)
            masks[i] = (mask >= torch.rand_like(mask.float())).to(torch.uint8)
            normalized_mask.append(mask.float().squeeze(0))

        if not is_batched:
            masks = masks[0]
            normalized_mask = normalized_mask[0]

        if return_normalized_mask:
            return masks, normalized_mask
        return masks

    def __call__(
        self,
        images,
        segmentation_maps: Optional[NonableImageInput] = None,
        return_patch_labels: bool = False,
        **kwargs
    ):
        if not isinstance(images, (list, tuple)):
            images = [images]
        images = [
            image.convert("RGB") if isinstance(image, PIL.Image.Image) else image
            for image in images
        ]
        if segmentation_maps is not None:
            if not isinstance(segmentation_maps, (list, tuple)):
                segmentation_maps = [segmentation_maps]
            mask_tensors = []
            normalized_masks: List[torch.Tensor] = []
            for i, mask in enumerate(segmentation_maps):
                if mask is None:
                    # if mask is not provided, create a zero tensor of the same size as the image
                    if isinstance(images[i], PIL.Image.Image):
                        W, H = images[i].size
                    elif isinstance(images[i], (np.ndarray, torch.Tensor)):
                        H, W = images[i].shape[:2]

                    mask_tensors.append(torch.zeros((H, W)))
                    normalized_masks.append(mask_tensors[-1])
                else:
                    mask, normalized_mask = self.convert_to_binary_masks(
                        mask, return_normalized_mask=True
                    )
                    mask_tensors.append(mask)
                    normalized_masks.append(normalized_mask)
        else:
            mask_tensors = None

        outputs = super().__call__(images, mask_tensors, **kwargs)

        if segmentation_maps is not None and return_patch_labels:
            patch_size = self.config.patch_size
            patch_labels = []
            pixel_values = outputs.pixel_values
            device = pixel_values[0].device
            for i, mask in enumerate(normalized_masks):
                H, W = pixel_values[i].shape[-2:]
                mask = F.interpolate(
                    mask[None, None, ...],
                    size=pixel_values[i].shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).view(H, W)
                # patch_size * (patch_size, W)
                rows = mask.split(patch_size, dim=0)
                patch_label_list = []
                for row in rows:
                    # patch_size * (patch_size, patch_size)
                    patches = row.split(patch_size, dim=1)
                    patch_label_list.append(
                        torch.tensor([patch.mean() for patch in patches], device=device)
                    )

                # (num_patches, num_patches)
                patch_labels.append(torch.stack(patch_label_list))
            outputs["patch_labels"] = torch.stack(patch_labels)

        return outputs

    def post_process_segmentation(
        self,
        outputs: LoupeUniversalOutput,
        target_sizes: Optional[List[Tuple[int, int]]] = None,
    ) -> List[torch.Tensor]:
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
        masks_queries_logits = outputs.masks_queries_logits
        class_queries_logits = outputs.class_queries_logits

        # Scale back to preprocessed image size - (384, 384) for all models
        masks_queries_logits = F.interpolate(
            masks_queries_logits, size=(384, 384), mode="bilinear", align_corners=False
        )

        masks_classes = class_queries_logits.softmax(dim=-1)
        masks_probs = (
            masks_queries_logits.sigmoid()
        )  # [batch_size, num_queries, height, width]

        # Semantic segmentation logits of shape (batch_size, num_classes + 1 (background class), height, width)
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
                resized_logits = F.interpolate(
                    segmentation[idx].unsqueeze(dim=0),
                    size=target_sizes[idx],
                    mode="bilinear",
                    align_corners=False,
                )
                semantic_map = resized_logits[0].argmax(dim=0)
                semantic_segmentation.append(semantic_map.to(dtype=torch.uint8))
        else:
            semantic_segmentation = segmentation.argmax(dim=1)
            # convert tensor to batch list
            semantic_segmentation = [
                semantic_segmentation[i].to(dtype=torch.uint8)
                for i in range(semantic_segmentation.shape[0])
            ]

        return semantic_segmentation
