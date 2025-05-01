from typing import Optional

from omegaconf import DictConfig, OmegaConf

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim


from deepspeed.ops.adam import DeepSpeedCPUAdam
from transformers import get_wsd_schedule, get_cosine_schedule_with_warmup

from metric import Metric
from models.loupe import LoupeModel


class LitModel(pl.LightningModule):
    def __init__(self, cfg: DictConfig, loupe: LoupeModel) -> None:
        super().__init__()
        self.cfg = cfg
        self.loupe = loupe
        self.model_config = loupe.config
        self.metric = Metric()
        self.val_outputs = []
        self.test_outputs = []

        if self.cfg.hparams.get("backbone_lr", 0) and self.cfg.model.freeze_backbone:
            raise ValueError(
                "backbone_lr is set to a specific value, but freeze_backbone is set to True. "
                "backbone_lr will be ignored."
            )

    def forward(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        labels: torch.Tensor,
        patch_labels: Optional[torch.Tensor] = None,
    ):
        """

        Forward pass for the model.

        Args:
            images (torch.Tensor): Input images with shape (N, C, H, W).
            masks (torch.Tensor): Input masks with shape (N, H, W).
            labels (torch.LongTensor): Labels with shape (N,), indicating whether the image is forgery.
                False means real, True means fake.
            patch_labels (Optional[torch.Tensor], optional): Patch labels with shape (N, num_patches, num_patches).
                Only used if config.enable_patch_cls is True. Defaults to None.
        """
        if self.cfg.stage == "cls":
            cls_output = self.loupe.cls_forward(
                pixel_values=images, labels=labels, patch_labels=patch_labels
            )
            return {
                "loss": cls_output.loss,
                "cls_logits": cls_output.logits,
            }
        if self.cfg.stage == "seg":
            seg_output = self.loupe.seg_forward(
                pixel_values=images, masks=masks, labels=labels
            )
            return {
                "loss": seg_output.loss,
                "cls_logits": seg_output.logits,
            }
        if self.cfg.stage == "test":
            cls_output = self.loupe.cls_forward(
                pixel_values=images, labels=labels, patch_labels=None
            )
            seg_output = self.loupe.seg_forward(
                pixel_values=images, masks=None, labels=labels
            )
            return {
                "cls_logits": cls_output.logits,
                "seg_logits": seg_output.logits,
            }

    def training_step(self, batch, batch_idx):
        outputs = self.forward(**batch)
        self.log_dict(outputs["loss"], sync_dist=True, prog_bar=True)

        return outputs["loss"]

    def configure_optimizers(self):
        def filter_decay_params(param_dict, **common_args):
            """filter parameters for optimizer, separate parameters by adding weight_decay or not"""
            non_decay_names = ["bias"]
            non_decay = [
                {
                    "params": [
                        p
                        for n, p in param_dict.items()
                        for name in non_decay_names
                        if name in n
                    ],
                    "weight_decay": 0.0,
                    **common_args,
                }
            ]

            decay = [
                {
                    "params": [
                        p
                        for n, p in param_dict.items()
                        for name in non_decay_names
                        if name not in n
                    ],
                    "weight_decay": self.cfg.hparams.weight_decay,
                    **common_args,
                }
            ]

            return [*non_decay, *decay]

        param_dict = {n: p for n, p in self.loupe.named_parameters() if p.requires_grad}
        optim_groups = []

        if self.cfg.hparams.backbone_lr:
            pe_param_dict = {n: p for n, p in param_dict.items() if "backbone" in n}
            pe_optim_groups = filter_decay_params(
                pe_param_dict, lr=self.cfg.hparams.backbone_lr
            )
            optim_groups.extend(pe_optim_groups)

            # filter out the parameters in pe_param_dict from param_dict
            param_dict = {
                n: p for n, p in param_dict.items() if n not in pe_param_dict.keys()
            }

        optim_groups.extend(filter_decay_params(param_dict, lr=self.cfg.hparams.lr))

        assert any(
            group["params"] is not None for group in optim_groups if "params" in group
        ), "No parameter to optimize."

        if "deepspeed" in self.cfg.strategy:
            optimizer = DeepSpeedCPUAdam(
                optim_groups,
                weight_decay=self.cfg.hparams.weight_decay,
            )
        else:
            optimizer = optim.AdamW(
                optim_groups,
                weight_decay=self.cfg.hparams.weight_decay,
            )

        step_batches = self.trainer.estimated_stepping_batches
        warmup_steps = self.cfg.hparams.warmup_step
        if isinstance(warmup_steps, float):
            warm_steps = warmup_steps * step_batches
        elif isinstance(warmup_steps, int):
            warm_steps = warmup_steps
        else:
            raise ValueError(
                f"the warm_steps should be int or float, but got {type(warmup_steps)}"
            )
        if self.cfg.hparams.scheduler == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warm_steps,
                num_training_steps=step_batches,
            )
        elif self.cfg.hparams.scheduler == "wsd":
            decay_steps = self.cfg.hparams.decay_step
            if isinstance(decay_steps, float):
                decay_steps = decay_steps * step_batches
            elif isinstance(decay_steps, int):
                decay_steps = decay_steps
            else:
                raise ValueError(
                    f"the decay_steps should be int or float, but got {type(decay_steps)}"
                )

            scheduler = get_wsd_schedule(
                optimizer,
                num_warmup_steps=warm_steps,
                num_decay_steps=decay_steps,
                num_training_steps=step_batches,
            )
        else:
            raise ValueError(
                f"the scheduler should be cosine or wsd, but got {self.cfg.hparams.scheduler}"
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs["loss"]["all"]
        preds = torch.sigmoid(outputs["cls_logits"]).squeeze(-1)
        targets = batch["labels"]
        self.val_outputs.append(
            {
                "val_loss": loss,
                "preds": preds,
                "targets": targets,
            }
        )
        return loss

    def test_step(self, batch, batch_idx):
        outputs = self(**batch)
        preds = torch.sigmoid(outputs["cls_logits"]).squeeze(-1)
        targets = batch["labels"]
        self.test_outputs.append(
            {
                "preds": preds,
                "targets": targets,
            }
        )

    def on_test_epoch_end(self):
        preds = torch.cat([o["preds"] for o in self.test_outputs])
        targets = torch.cat([o["targets"] for o in self.test_outputs])
        best_threshold_f1, best_f1 = self.metric.find_best_threshold(
            self.metric.compute_f1, preds, targets
        )
        auc = self.metric.compute_auc(preds, targets)

        self.log("f1", best_f1, prog_bar=True, sync_dist=True)
        self.log("f1_thres", best_threshold_f1, prog_bar=True, sync_dist=True)
        self.log("auc", auc, prog_bar=True, sync_dist=True)
        self.test_outputs.clear()

    def on_validation_epoch_end(self):
        avg_loss = torch.stack([o["val_loss"] for o in self.val_outputs]).mean()
        preds = torch.cat([o["preds"] for o in self.val_outputs])
        targets = torch.cat([o["targets"] for o in self.val_outputs])
        best_threshold_f1, best_f1 = self.metric.find_best_threshold(
            self.metric.compute_f1, preds, targets
        )
        auc = self.metric.compute_auc(preds, targets)

        self.log("val_loss", avg_loss, prog_bar=True, sync_dist=True)
        self.log("f1", best_f1, prog_bar=True, sync_dist=True)
        self.log("f1_thres", best_threshold_f1, prog_bar=True, sync_dist=True)
        self.log("auc", auc, prog_bar=True, sync_dist=True)
        self.val_outputs.clear()

    def on_save_checkpoint(self, checkpoint):
        full_state_dict = self.state_dict()

        trainable_state_dict = {
            name: param
            for name, param in full_state_dict.items()
            if self.get_parameter(name).requires_grad
        }

        checkpoint["state_dict"] = trainable_state_dict
        return checkpoint

    def on_load_checkpoint(self, checkpoint):
        trainable_state_dict = checkpoint["state_dict"]
        self.load_state_dict(trainable_state_dict, strict=False)
