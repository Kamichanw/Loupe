from typing import Callable, List
import torch
import numpy as np
import os
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from torchmetrics.classification import BinaryJaccardIndex


class Metric:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def compute_f1(
        self, preds: torch.Tensor, targets: torch.Tensor, threshold=None
    ) -> float:
        threshold = self.threshold if threshold is None else threshold
        preds = (preds > threshold).int()
        targets = targets.int()

        f1 = f1_score(
            targets.cpu().numpy(),
            preds.cpu().numpy(),
            average="binary",
            zero_division=1,
        )
        return f1

    def compute_auc(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
        preds = preds.detach().cpu().numpy()
        targets = targets.int().cpu().numpy()
        auc = roc_auc_score(targets, preds, average="macro")
        return auc

    def compute_accuracy(
        self, preds: torch.Tensor, targets: torch.Tensor, threshold=None
    ) -> float:
        threshold = self.threshold if threshold is None else threshold
        preds = (preds > threshold).int()
        targets = targets.int()

        acc = accuracy_score(targets.cpu().numpy(), preds.cpu().numpy())
        return acc

    def compute_iou(self, preds: List[torch.Tensor], targets: List[torch.Tensor]) -> float:
        """
        Compute average IoU across a list of predicted and ground-truth masks.

        Args:
            preds (List[Tensor]): List of predicted masks, each of shape (H_i, W_i)
            targets (List[Tensor]): List of ground-truth masks, each of shape (H_i, W_i)

        Returns:
            float: Average IoU score.
        """
        hist = torch.zeros((1, 1), device=preds[0].device)

        for pred, target in zip(preds, targets):
            assert pred.flatten().shape == target.flatten().shape
            pred_bin = pred.int()
            target_bin = target.int()

            mask = (target_bin >= 0) & (target_bin < 1)
            hist += torch.bincount(
                (target_bin[mask].long() * 1 + pred_bin[mask].long()), minlength=1
            ).reshape(1, 1)

        iou = torch.diag(hist) / (hist.sum(dim=1) + hist.sum(dim=0) - torch.diag(hist))
        miou = torch.nanmean(iou)

        return miou.item()

    def find_best_threshold(
        self,
        func: Callable[[torch.Tensor, torch.Tensor], float],
        preds: torch.Tensor,
        targets: torch.Tensor,
        step=0.05,
    ):
        """
        Sweep thresholds from 0 to 1 and find the best metric score.
        Returns:
            best_threshold (float), best_metric (float)
        """
        preds = preds.detach().cpu()
        targets = targets.detach().cpu()

        thresholds = np.arange(0.0, 1.0 + step, step)
        best_metric = -1
        best_threshold = 0.5

        for t in thresholds:
            metric = func(preds, targets, threshold=t)
            if metric > best_metric:
                best_metric = metric
                best_threshold = t

        return best_threshold, best_metric

    def __call__(
        self,
        preds: torch.Tensor = None,
        targets: torch.Tensor = None,
        mask_preds: torch.Tensor = None,
        mask_labels: torch.Tensor = None,
    ) -> dict:
        """
        Compute metrics. If mask_preds and mask_labels are provided, compute mask IOU too.
        """
        result = {}
        if preds is not None and targets is not None:
            result["f1"] = self.compute_f1(preds, targets)
            result["auc"] = self.compute_auc(preds, targets)
            result["accuracy"] = self.compute_accuracy(preds, targets)

        if mask_preds is not None and mask_labels is not None:
            mask_preds = mask_preds.flatten()
            mask_labels = mask_labels.flatten()
            result["mask_iou"] = self.compute_iou(mask_preds, mask_labels)

        return result
