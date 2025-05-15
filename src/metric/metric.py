import os
import torch
import numpy as np
import evaluate
from pathlib import Path
from typing import Callable, List, overload
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score


class Metric:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        parent_dir = Path(__file__).resolve().parent
        self.iou_metric = evaluate.load(os.path.join(parent_dir, "mean_iou.py"))

    @overload
    def compute_f1(
        self, preds: torch.Tensor, targets: torch.Tensor, threshold=None
    ) -> float:
        """
        Compute F1 score for binary classification.
        """
        ...

    @overload
    def compute_f1(
        self, preds: List[torch.Tensor], targets: List[torch.Tensor], threshold=None
    ) -> float:
        """
        Compute F1 score for mask predictions.
        """
        ...

    def compute_f1(
        self,
        preds: torch.Tensor | List[torch.Tensor],
        targets: torch.Tensor | List[torch.Tensor],
        threshold=None,
    ) -> float:
        """
        Compute F1 score for binary classification or mask predictions.
        If preds and targets are tensors, compute F1 score for binary classification.
        If preds and targets are lists of tensors, compute F1 score for mask predictions.
        """
        threshold = self.threshold if threshold is None else threshold
        if isinstance(preds, list) and isinstance(targets, list):
            # for mask predictions
            preds, targets = self._postprocess_masks_and_labels(preds, targets)
            preds = torch.cat([pred.view(-1) for pred in preds], dim=0)
            targets = torch.cat([target.view(-1) for target in targets], dim=0)
        else:
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
        preds = preds.detach().float().cpu().numpy()
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

    def _postprocess_masks_and_labels(
        self,
        preds: List[torch.Tensor],
        targets: List[torch.Tensor],
    ):
        """
        Revert the mask predictions and skip real images.
        """
        predictions = []
        references = []
        for pred, target in zip(preds, targets):
            # skip real images
            if not torch.all(target == 0):
                # in predictions, 0 is for forged region, 1 is for background (no object class)
                # so we need to invert the mask
                pred = torch.where(pred == 0, 1, 0)
                predictions.append(pred.to(dtype=torch.uint8))
                references.append(target)
        return predictions, references

    def compute_iou(
        self, preds: List[torch.Tensor], targets: List[torch.Tensor]
    ) -> float:
        predictions, references = self._postprocess_masks_and_labels(preds, targets)
        return self.iou_metric.compute(
            predictions=predictions,
            references=references,
            num_labels=2,
            ignore_index=255,
        )["mean_iou"]

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
