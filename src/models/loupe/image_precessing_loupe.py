import random
from typing import Dict, List, Optional, Tuple, Union, Unpack
import PIL
import PIL.Image
import numpy as np
import torch
from torchvision import transforms as T
from transformers.image_processing_utils_fast import (
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
    BatchFeature,
)
from transformers.models.mask2former import Mask2FormerImageProcessor
from transformers.image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    PILImageResampling,
    ImageInput
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


class LoupeImageProcessor(Mask2FormerImageProcessor):
    def __init__(
        self,
        config: LoupeConfig,
        size: Dict[str, int] = None,
        do_resize=True,
        do_rescale=True,
        do_normalize=True,
        do_reduce_labels=True,
        image_mean: Union[float, List[float]] = None,
        image_std: Union[float, List[float]] = None,
        ignore_index: Optional[int] = None,
        **kwargs
    ):
        super().__init__(
            size=size or {"height": config.image_size, "width": config.image_size},
            do_resize=do_resize,
            do_rescale=do_rescale,
            do_normalize=do_normalize,
            do_reduce_labels=do_reduce_labels,
            image_mean=image_mean or IMAGENET_STANDARD_MEAN,
            image_std=image_std or IMAGENET_STANDARD_STD,
            ignore_index=ignore_index or 255,
            **kwargs
        )
        self.config = config
    
    def convert_to_binary_masks(self, masks: ImageInput, return_normalized_mask: bool = False):
        """ convert masks to binary mask with 0 and 1 """
        is_batched = isinstance(masks, (list, tuple))
        if not is_batched:
            masks = [masks]
        normalized_mask = []
        for i, mask in enumerate(masks):
            if isinstance(mask, PIL.Image.Image):
                mask = np.array(mask.convert("L"))
            if isinstance(mask, np.ndarray):
                mask = torch.tensor(mask, dtype=torch.uint8)
            min_val, max_val = mask.min(), mask.max()
            if max_val - min_val > 0:
                mask = (mask - min_val) / (max_val - min_val)
            else:
                mask = torch.zeros_like(mask, dtype=torch.uint8)
            masks[i] = (mask >= torch.rand_like(mask)).to(torch.uint8)
            normalized_mask.append(mask.squeeze(0))
        
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
                    mask, normalized_mask = self.convert_to_binary_masks(mask, return_normalized_mask=True)
                    mask_tensors.append(mask)
                    normalized_masks.append(normalized_mask)
        else:
            mask_tensors = None

        outputs = super().__call__(images, mask_tensors, **kwargs)

        if segmentation_maps is not None and return_patch_labels:
            patch_size = self.config.patch_size
            patch_labels = []
            device = outputs.pixel_values[0].device
            for mask in normalized_masks:
                H, W = mask.shape
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
            outputs["patch_labels"] = patch_labels

        return outputs

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
