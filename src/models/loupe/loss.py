from typing import Dict, List, Literal, Optional, Tuple
import numpy as np
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerLoss,
    pair_wise_dice_loss,
    pair_wise_sigmoid_cross_entropy_loss,
    Mask2FormerHungarianMatcher,
    sample_point,
)
from transformers.utils import is_scipy_available
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.loupe.configuration_loupe import LoupeConfig

if is_scipy_available():
    from scipy.optimize import linear_sum_assignment


class Poly1FocalLoss(nn.Module):
    """
    Poly1FocalLoss for multi-class classification (supports both hard and soft labels).

    Combines Cross-Entropy Loss with Focal Loss modulation and a polynomial penalty term.
    Supports hard labels (class indices) and soft labels (class probabilities).

    Args:
        epsilon (float): Coefficient for the poly1 penalty term.
        gamma (float): Focusing parameter for focal term.
        alpha (float): Class balancing weight for positive class.
        reduction (str): 'mean' | 'sum' | 'none' (default: 'mean').

    Inputs:
        logits (Tensor): Raw model outputs before softmax. Shape: [B, C]
        targets (Tensor):
            - If dtype == long: hard labels, shape [B]
            - If dtype == float: soft labels, shape [B, C]

    Returns:
        loss (Tensor): Scalar if reduced, else per-sample vector of shape [B]
    """

    def __init__(self, epsilon=1.0, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        p = torch.sigmoid(logits)
        if labels.dtype == torch.long:
            num_classes = logits.shape[1]
            # if labels are of shape [N]
            # convert to one-hot tensor of shape [N, num_classes]
            if labels.ndim == 1:
                labels = F.one_hot(labels, num_classes=num_classes)

            # if labels are of shape [N, ...] e.g. segmentation task
            # convert to one-hot tensor of shape [N, num_classes, ...]
            else:
                labels = (
                    F.one_hot(labels.unsqueeze(1), num_classes)
                    .transpose(1, -1)
                    .squeeze_(-1)
                )
        else:
            if labels.shape != logits.shape:
                raise ValueError(
                    f"Shape mismatch: logits {logits.shape} and labels {labels.shape}."
                )

        labels = labels.to(device=logits.device, dtype=logits.dtype)

        ce_loss = F.binary_cross_entropy_with_logits(
            input=logits,
            target=labels,
            reduction="none",
            weight=self.alpha * labels + (1 - self.alpha) * (1 - labels),
        )
        pt = labels * p + (1 - labels) * (1 - p)
        FL = ce_loss * ((1 - pt) ** self.gamma)

        poly1 = FL + self.epsilon * torch.pow(1 - pt, self.gamma + 1)

        if self.reduction == "mean":
            poly1 = poly1.mean()
        elif self.reduction == "sum":
            poly1 = poly1.sum()

        return poly1


class TverskyLoss(nn.Module):
    """
    Tversky loss for binary masks.
    Args:
        alpha (float): Weight for false positives.
        beta (float): Weight for false negatives.
        reduction (str): 'batchmean' | 'sum' | 'none' (default: 'batchmean').
    """

    def __init__(self, alpha=0.5, beta=0.5, reduction="batchmean"):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs (Tensor): Raw logits from the model.
            labels (Tensor): Ground-truth masks (binary). Same shape as inputs.
        Returns:
            Tensor: Scalar loss.
        """
        probs = inputs.sigmoid().flatten(1)
        labels = labels.flatten(1).float()

        TP = (probs * labels).sum(-1)
        FP = (probs * (1 - labels)).sum(-1)
        FN = ((1 - probs) * labels).sum(-1)

        tversky = (TP + 1e-6) / (TP + self.alpha * FP + self.beta * FN + 1e-6)
        loss = 1.0 - tversky

        if self.reduction == "batchmean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss


class LoupeClsLoss(nn.Module):
    def __init__(
        self,
        config: LoupeConfig,
    ):
        super(LoupeClsLoss, self).__init__()
        self.patch_wise_criterion = Poly1FocalLoss(
            epsilon=1.0,
            alpha=config.patch_forge_weight,
            gamma=2.0,
        )
        self.cls_criterion = Poly1FocalLoss(
            epsilon=1.0,
            alpha=config.cls_forge_weight,
            gamma=2.0,
        )

    def forward(
        self,
        cls_logits: Optional[torch.Tensor] = None,
        cls_labels: Optional[torch.Tensor] = None,
        patch_logits: Optional[torch.Tensor] = None,
        patch_labels: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            cls_logits: Tensor of shape [N, 1], raw logits of real or fake classification.
            cls_labels: Tensor of shape [N, 1], whether the image is forged or not.
            patch_logits: Tensor of shape [N, num_patches], raw logits (before sigmoid).
            patch_labels: Tensor of shape [N, num_patches], containing (fake_prob).
        Returns:
            Loss value (scalar or tensor depending on reduction).
        """
        loss = None
        # calculate regular classification loss
        if cls_logits is not None and cls_labels is not None:
            # move labels to correct device to enable model parallelism
            cls_labels = cls_labels.to(cls_logits.device, dtype=cls_logits.dtype)
            loss = self.cls_criterion(
                cls_logits.view(-1),
                cls_labels.view(-1),
            )

        if patch_logits is not None and patch_labels is not None:
            # calculate patch classification loss
            patch_labels = patch_labels.to(
                patch_logits.device, dtype=patch_logits.dtype
            )
            patch_wise_loss = self.patch_wise_criterion(
                patch_logits.view(-1),
                patch_labels.view(-1),
            )
            if loss is not None:
                loss = loss + patch_wise_loss
            else:
                loss = patch_wise_loss

        return loss


class LoupeHungarianMatcher(Mask2FormerHungarianMatcher):
    def __init__(
        self,
        cost_class: float = 1.0,
        cost_mask: float = 1.0,
        cost_dice: float = 1.0,
        num_points: int = 12544,
    ):
        super().__init__(
            cost_class=cost_class,
            cost_mask=cost_mask,
            cost_dice=cost_dice,
            num_points=num_points,
        )

    @torch.no_grad()
    def forward(
        self,
        masks_queries_logits: torch.Tensor,
        class_queries_logits: torch.Tensor,
        mask_labels: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> List[Tuple[torch.Tensor]]:
        """
        Params:
            masks_queries_logits (`torch.Tensor`):
                A tensor of dim `batch_size, num_queries, num_labels` with the classification logits.
            class_queries_logits (`torch.Tensor`):
                A tensor of dim `batch_size, num_queries, height, width` with the predicted masks.
            class_labels (`torch.Tensor`):
                A tensor of dim `num_target_boxes` (where num_target_boxes is the number of ground-truth objects in the
                target) containing the class labels.
            mask_labels (`torch.Tensor`):
                A tensor of dim `num_target_boxes, height, width` containing the target masks.

        Returns:
            matched_indices (`List[Tuple[Tensor]]`): A list of size batch_size, containing tuples of (index_i, index_j)
            where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected labels (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes).
        """
        indices: List[Tuple[np.array]] = []

        # iterate through batch size
        batch_size = masks_queries_logits.shape[0]
        for i in range(batch_size):
            pred_probs = class_queries_logits[i].softmax(-1)
            pred_mask = masks_queries_logits[i]

            # Compute the classification cost. Contrary to the loss, we don't use the NLL, but approximate it in 1 - proba[target class]. The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -pred_probs[:, class_labels[i]]
            target_mask = mask_labels[i].to(pred_mask)
            target_mask = target_mask[:, None]
            pred_mask = pred_mask[:, None]

            # Sample ground truth and predicted masks
            point_coordinates = torch.rand(
                1, self.num_points, 2, device=pred_mask.device, dtype=pred_mask.dtype
            )

            target_coordinates = point_coordinates.repeat(target_mask.shape[0], 1, 1)
            target_mask = sample_point(
                target_mask, target_coordinates, align_corners=False
            ).squeeze(1)

            pred_coordinates = point_coordinates.repeat(pred_mask.shape[0], 1, 1)
            pred_mask = sample_point(
                pred_mask, pred_coordinates, align_corners=False
            ).squeeze(1)

            # compute the cross entropy loss between each mask pairs -> shape (num_queries, num_labels)
            cost_mask = pair_wise_sigmoid_cross_entropy_loss(pred_mask, target_mask)
            # Compute the dice loss betwen each mask pairs -> shape (num_queries, num_labels)
            cost_dice = pair_wise_dice_loss(pred_mask, target_mask)
            # final cost matrix
            cost_matrix = (
                self.cost_mask * cost_mask
                + self.cost_class * cost_class
                + self.cost_dice * cost_dice
            )
            # eliminate infinite values in cost_matrix to avoid the error ``ValueError: cost matrix is infeasible``
            cost_matrix = torch.minimum(cost_matrix, torch.tensor(1e10))
            cost_matrix = torch.maximum(cost_matrix, torch.tensor(-1e10))
            cost_matrix = torch.nan_to_num(cost_matrix, 0)
            # do the assigmented using the hungarian algorithm in scipy
            assigned_indices: Tuple[np.array] = linear_sum_assignment(
                cost_matrix.to(torch.float).cpu()
            )
            indices.append(assigned_indices)

        # It could be stacked in one tensor
        matched_indices = [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]
        return matched_indices


# there are too many bugs in transformers implementation
# so we have to rewrite the loss some parts
class LoupeSegLoss(Mask2FormerLoss):
    """
    We modified the original Mask2FormerLoss with the following changes:
    1. Use Poly1FocalLoss instead of CrossEntropyLoss for class labels.
    2.
    """

    def __init__(self, config: LoupeConfig):
        self.weight_dict = {
            "loss_cross_entropy": config.mask2former_config.class_weight,
            "loss_mask": config.mask2former_config.mask_weight,
            "loss_dice": config.mask2former_config.dice_weight,
        }
        super().__init__(config.mask2former_config, self.weight_dict)

        self.matcher = LoupeHungarianMatcher(
            cost_mask=config.mask2former_config.mask_weight,
            cost_dice=config.mask2former_config.dice_weight,
            num_points=config.mask2former_config.train_num_points,
        )
        self.pixel_mask_criterion = Poly1FocalLoss(
            epsilon=config.pixel_poly_epsilon,
            alpha=config.pixel_forge_weight,
            gamma=2.0,
        )
        self.pixel_tversky_criterion = TverskyLoss(
            alpha=config.tversky_alpha,
            beta=1 - config.tversky_alpha,
        )

        self.use_auxiliary_loss = config.mask2former_config.use_auxiliary_loss

    def loss_masks(
        self,
        masks_queries_logits: torch.Tensor,
        mask_labels: List[torch.Tensor],
        indices: Tuple[np.array],
        num_masks: int,
    ) -> Dict[str, torch.Tensor]:
        """Compute the losses related to the masks using sigmoid_cross_entropy_loss and dice loss.

        Args:
            masks_queries_logits (`torch.Tensor`):
                A tensor of shape `(batch_size, num_queries, height, width)`.
            mask_labels (`torch.Tensor`):
                List of mask labels of shape `(labels, height, width)`.
            indices (`Tuple[np.array])`:
                The indices computed by the Hungarian matcher.
            num_masks (`int)`:
                The number of masks, used for normalization.

        Returns:
            losses (`Dict[str, Tensor]`): A dict of `torch.Tensor` containing two keys:
            - **loss_mask** -- The loss computed using sigmoid cross entropy loss on the predicted and ground truth.
              masks.
            - **loss_dice** -- The loss computed using dice loss on the predicted on the predicted and ground truth,
              masks.
        """
        num_masks = int(num_masks)
        src_idx = self._get_predictions_permutation_indices(indices)
        tgt_idx = self._get_targets_permutation_indices(indices)
        # shape (batch_size * num_queries, height, width)
        pred_masks = masks_queries_logits[src_idx]
        # shape (batch_size, num_queries, height, width)
        # pad all and stack the targets to the num_labels dimension
        target_masks, _ = self._pad_images_to_max_in_batch(mask_labels)
        target_masks = target_masks[tgt_idx]

        target_masks = F.interpolate(target_masks[:, None], size=pred_masks.shape[-2:])
        losses = {}
        if self.weight_dict["loss_mask"] > 0:
            losses["loss_mask"] = self.pixel_mask_criterion(
                pred_masks.view(num_masks, -1), target_masks.view(num_masks, -1)
            )
        if self.weight_dict["loss_dice"] > 0:
            losses["loss_dice"] = self.pixel_tversky_criterion(pred_masks, target_masks)

        del pred_masks
        del target_masks
        return losses

    def forward(
        self,
        masks_queries_logits: torch.Tensor,
        class_queries_logits: torch.Tensor,
        mask_labels: List[torch.Tensor],
        class_labels: List[torch.Tensor],
        auxiliary_predictions: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        This performs the loss computation.

        Args:
            masks_queries_logits (`torch.Tensor`):
                A tensor of shape `(batch_size, num_queries, height, width)`.
            class_queries_logits (`torch.Tensor`):
                A tensor of shape `(batch_size, num_queries, num_labels)`.
            mask_labels (`torch.Tensor`):
                List of mask labels of shape `(labels, height, width)`.
            class_labels (`List[torch.Tensor]`):
                List of class labels of shape `(labels)`.
            auxiliary_predictions (`Dict[str, torch.Tensor]`, *optional*):
                if `use_auxiliary_loss` was set to `true` in [`Mask2FormerConfig`], then it contains the logits from
                the inner layers of the Mask2FormerMaskedAttentionDecoder.

        Returns:
            losses (`Dict[str, Tensor]`): A dict of `torch.Tensor` containing three keys:
            - **loss_cross_entropy** -- The loss computed using cross entropy on the predicted and ground truth labels.
            - **loss_mask** -- The loss computed using sigmoid cross_entropy loss on the predicted and ground truth
              masks.
            - **loss_dice** -- The loss computed using dice loss on the predicted on the predicted and ground truth
              masks.
            if `use_auxiliary_loss` was set to `true` in [`Mask2FormerConfig`], the dictionary contains additional
            losses for each auxiliary predictions.
        """

        # retrieve the matching between the outputs of the last layer and the labels
        indices = self.matcher(
            masks_queries_logits, class_queries_logits, mask_labels, class_labels
        )
        # compute the average number of target masks for normalization purposes
        num_masks = self.get_num_masks(class_labels, device=class_labels[0].device)
        # get all the losses
        losses: Dict[str, torch.Tensor] = {
            **self.loss_masks(masks_queries_logits, mask_labels, indices, num_masks),
            **self.loss_labels(class_queries_logits, class_labels, indices),
        }
        # in case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if self.use_auxiliary_loss and auxiliary_predictions is not None:
            for idx, aux_outputs in enumerate(auxiliary_predictions):
                masks_queries_logits = aux_outputs["masks_queries_logits"]
                class_queries_logits = aux_outputs["class_queries_logits"]
                loss_dict = self.forward(
                    masks_queries_logits,
                    class_queries_logits,
                    mask_labels,
                    class_labels,
                )
                loss_dict = {f"{key}_{idx}": value for key, value in loss_dict.items()}
                losses.update(loss_dict)

        return losses

    def sample_points_using_uncertainty(
        self,
        logits: torch.Tensor,
        uncertainty_function,
        num_points: int,
        oversample_ratio: int,
        importance_sample_ratio: float,
    ) -> torch.Tensor:
        """
        This function is meant for sampling points in [0, 1] * [0, 1] coordinate space based on their uncertainty. The
        uncertainty is calculated for each point using the passed `uncertainty function` that takes points logit
        prediction as input.

        Args:
            logits (`float`):
                Logit predictions for P points.
            uncertainty_function:
                A function that takes logit predictions for P points and returns their uncertainties.
            num_points (`int`):
                The number of points P to sample.
            oversample_ratio (`int`):
                Oversampling parameter.
            importance_sample_ratio (`float`):
                Ratio of points that are sampled via importance sampling.

        Returns:
            point_coordinates (`torch.Tensor`):
                Coordinates for P sampled points.
        """

        num_boxes = logits.shape[0]
        num_points_sampled = int(num_points * oversample_ratio)

        # Get random point coordinates
        point_coordinates = torch.rand(
            num_boxes, num_points_sampled, 2, device=logits.device, dtype=logits.dtype
        )
        # Get sampled prediction value for the point coordinates
        point_logits = sample_point(logits, point_coordinates, align_corners=False)
        # Calculate the uncertainties based on the sampled prediction values of the points
        point_uncertainties = uncertainty_function(point_logits)

        num_uncertain_points = int(importance_sample_ratio * num_points)
        num_random_points = num_points - num_uncertain_points

        idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
        shift = num_points_sampled * torch.arange(
            num_boxes, dtype=torch.long, device=logits.device
        )
        idx += shift[:, None]
        point_coordinates = point_coordinates.view(-1, 2)[idx.view(-1), :].view(
            num_boxes, num_uncertain_points, 2
        )

        if num_random_points > 0:
            point_coordinates = torch.cat(
                [
                    point_coordinates,
                    torch.rand(
                        num_boxes,
                        num_random_points,
                        2,
                        device=logits.device,
                        dtype=logits.dtype,
                    ),
                ],
                dim=1,
            )
        return point_coordinates
