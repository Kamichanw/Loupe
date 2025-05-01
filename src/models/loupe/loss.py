from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F


# reference: https://github.com/abhuse/polyloss-pytorch/blob/main/polyloss.py
class Poly1FocalLoss(nn.Module):
    """
    Poly1FocalLoss for binary classification tasks (real vs fake), supporting soft labels.

    Args:
        epsilon (float): Poly loss epsilon term coefficient.
        alpha (float): Class balancing factor (weight for positive samples).
        gamma (float): Focusing parameter for focal loss.
        reduction: Literal["mean", "sum", "none"] = "mean",
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        alpha: float = 0.9,
        gamma: float = 2.0,
    ):
        super(Poly1FocalLoss, self).__init__()
        assert 0 <= alpha <= 1, "alpha should be between 0 and 1."
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = "mean"

    def forward(
        self,
        logits,
        labels,
        **kwargs,
    ):
        """
        Args:
            logits: Tensor of shape [N, num_patches], raw logits (before sigmoid).
            labels: Tensor of shape [N, num_patches], containing (fake_prob).
            kwargs: Additional arguments passed to binary_cross_entropy.
        Returns:
            Loss value (scalar or tensor depending on reduction).
        """
        p = torch.sigmoid(logits)

        assert (
            p.shape == labels.shape
        ), f"Expected logits and labels to have the same shape, got {logits.shape} and {labels.shape}."

        ce_loss = F.binary_cross_entropy_with_logits(
            input=logits, target=labels, reduction="none", **kwargs
        )

        pt = labels * p + (1 - labels) * (1 - p)
        FL = ce_loss * ((1 - pt) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
            FL = alpha_t * FL

        poly1 = FL + self.epsilon * torch.pow(1 - pt, self.gamma + 1)

        if self.reduction == "mean":
            poly1 = poly1.mean()
        elif self.reduction == "sum":
            poly1 = poly1.sum()

        return poly1


class LoupeClsLoss(nn.Module):
    def __init__(
        self,
        epsilon: float = 1.0,
        alpha: float = 0.9,
        gamma: float = 2.0,
    ):
        super(LoupeClsLoss, self).__init__()
        self.patch_wise_criterion = Poly1FocalLoss(
            epsilon=epsilon, alpha=alpha, gamma=gamma
        )
        self.cls_criterion = nn.BCEWithLogitsLoss()

    def forward(
        self,
        cls_logits=None,
        cls_labels=None,
        patch_logits=None,
        patch_labels=None,
        masks=None,
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
