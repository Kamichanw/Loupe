import re
import pytorch_lightning as pl
import torch
import torch.optim as optim

from typing import Optional
from PIL import Image
from loguru import logger
from omegaconf import DictConfig
from deepspeed.ops.adam import DeepSpeedCPUAdam
from transformers import get_wsd_schedule, get_cosine_schedule_with_warmup

from metric import Metric
from models.loupe import LoupeModel
from models.loupe.image_precessing_loupe import LoupeImageProcessor
from models.loupe.modeling_loupe import LoupeUniversalOutput

torch.serialization.add_safe_globals([set])


class LitModel(pl.LightningModule):
    def __init__(self, cfg: DictConfig, loupe: LoupeModel) -> None:
        super().__init__()
        self.cfg = cfg
        self.loupe = loupe
        self.model_config = loupe.config
        self.processor = LoupeImageProcessor(self.model_config)
        self.metric = Metric()
        self.val_outputs = []

        if getattr(self.cfg.hparams, "backbone_lr", None) and getattr(
            self.cfg.model, "freeze_backbone", None
        ):
            logger.warning(
                "backbone_lr is set to a specific value, but freeze_backbone is set to True. "
                "backbone_lr will be ignored."
            )

        for ckpt_path in self.cfg.ckpt.checkpoint_paths:
            if ckpt_path.endswith(".safetensors"):
                from safetensors.torch import load_file

                state_dict = load_file(ckpt_path)
            elif ckpt_path.endswith(".pt") or ckpt_path.endswith(".pth"):
                state_dict = torch.load(ckpt_path)
            else:
                raise ValueError(
                    f"Unsupported checkpoint format: {ckpt_path}. "
                    "Please use .safetensors, .pt or .pth format."
                )
            logger.info(f"Loading checkpoint from {ckpt_path}")
            self.load_state_dict(
                state_dict=state_dict,
                strict=False,
            )

    def forward(
        self,
        pixel_values: torch.Tensor,
        mask_labels: Optional[torch.Tensor] = None,
        pixel_mask: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None,
        patch_labels: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> LoupeUniversalOutput:
        """
        Forward pass for the model.

        Args:
            pixel_values (torch.Tensor): Input images with shape (N, C, H, W).
            mask_labels (torch.Tensor): Input masks with shape (N, H, W).
            class_labels (torch.LongTensor): Labels with shape (N, 0 or 1). If the ith image is forgery, class_labels[i] == tensor([1]),
                else class_labels[i] == tensor([]).
            patch_labels (Optional[torch.Tensor], optional): Patch labels with shape (N, num_patches, num_patches).
                Only used if config.enable_patch_cls is True. Defaults to None.
            labels (Optional[torch.Tensor], optional): Labels with shape (N,), which is used to classify the image.
        """
        return self.loupe(
            pixel_values=pixel_values,
            mask_labels=mask_labels,
            pixel_mask=pixel_mask,
            class_labels=class_labels,
            patch_labels=patch_labels,
            labels=labels,
        )

    def training_step(self, batch, batch_idx):
        outputs = self.forward(**batch)

        loss_dict = {"total_loss": outputs.loss}
        if "cls" in self.cfg.stage.name:
            loss_dict["cls_loss"] = outputs.loss_dict["cls"]["loss"]
        if "seg" in self.cfg.stage.name:
            loss_dict["seg_loss"] = outputs.loss_dict["seg"].pop("loss")
            loss_dict.update(outputs.loss_dict["seg"])

        for key, value in loss_dict.items():
            is_auxiliary_loss = re.search(r"_\d+$", key) is not None
            # hide auxiliary loss from mask2former in the progress bar
            self.log(key, value, prog_bar=not is_auxiliary_loss, sync_dist=True)

        return outputs.loss

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

        def set_hparam(param_dict, pattern: str, **common_args):
            """set hparams for params that contains the pattern, and return the remaining param dict"""
            selected_params = {n: p for n, p in param_dict.items() if pattern in n}
            pe_optim_groups = filter_decay_params(selected_params, **common_args)
            optim_groups.extend(pe_optim_groups)

            # filter out the parameters in selected_params from param_dict
            return {
                n: p for n, p in param_dict.items() if n not in selected_params.keys()
            }

        param_dict = {n: p for n, p in self.loupe.named_parameters() if p.requires_grad}
        optim_groups = []

        if self.cfg.hparams.backbone_lr:
            param_dict = set_hparam(
                param_dict, "backbone", lr=self.cfg.hparams.backbone_lr
            )

        if getattr(self.cfg.hparams, "cls_lr", None):
            param_dict = set_hparam(
                param_dict, "classifier", lr=self.cfg.hparams.cls_lr
            )

        if getattr(self.cfg.hparams, "seg_lr", None):
            param_dict = set_hparam(param_dict, "segmentor", lr=self.cfg.hparams.seg_lr)

        if param_dict:
            optim_groups.extend(filter_decay_params(param_dict, lr=self.cfg.hparams.lr))

        assert any(
            group["params"] is not None for group in optim_groups if "params" in group
        ), "No parameter to optimize."

        if self.cfg.strategy and "deepspeed" in self.cfg.strategy:
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
        masks = batch.pop("masks")
        with torch.no_grad():
            outputs = self.forward(**batch)
        val_output = {
            "val_loss": outputs.loss,
        }
        if outputs.loss_dict["cls"] is not None:
            val_output.update(
                {
                    "val_cls_loss": outputs.loss_dict["cls"]["loss"],
                    "cls_preds": torch.sigmoid(outputs.cls_logits).squeeze(-1),
                    "cls_targets": batch["labels"],
                }
            )
        if outputs.loss_dict["seg"] is not None:
            target_sizes = [
                (mask.shape[0], mask.shape[1]) for mask in masks
            ]  #  (H_i, W_i)
            val_output.update(
                {
                    "val_seg_loss": outputs.loss_dict["seg"]["loss"],
                    "seg_preds": self.processor.post_process_segmentation(
                        outputs, target_sizes=target_sizes
                    ),
                    "seg_targets": masks,
                }
            )
        self.val_outputs.append(val_output)
        return outputs.loss

    def on_validation_epoch_end(self):
        metric_dict = {
            "val_loss": torch.stack([o["val_loss"] for o in self.val_outputs]).mean()
        }
        if "val_cls_loss" in self.val_outputs[0]:
            preds = torch.cat([o["cls_preds"] for o in self.val_outputs])
            targets = torch.cat([o["cls_targets"] for o in self.val_outputs])
            auc = self.metric.compute_auc(preds, targets)
            metric_dict.update(
                {
                    "val_cls_loss": torch.stack(
                        [o["val_cls_loss"] for o in self.val_outputs]
                    ).mean(),
                    "auc": auc,
                }
            )
        if "val_seg_loss" in self.val_outputs[0]:
            preds = [p for o in self.val_outputs for p in o["seg_preds"]]
            targets = [t for o in self.val_outputs for t in o["seg_targets"]]
            iou = self.metric.compute_iou(preds, targets)
            f1 = self.metric.compute_f1(preds, targets)
            metric_dict.update(
                {
                    "val_seg_loss": torch.stack(
                        [o["val_seg_loss"] for o in self.val_outputs]
                    ).mean(),
                    "iou": iou,
                    "f1": f1,
                }
            )

        if self.cfg.stage.name == "cls_seg":
            metric_dict["overall"] = (
                metric_dict["auc"] + metric_dict["iou"] + metric_dict["f1"]
            ) / 3

        self.log_dict(
            metric_dict,
            prog_bar=True,
            sync_dist=True,
        )
        self.val_outputs.clear()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        return self.on_validation_epoch_end()

    def predict_step(self, batch, batch_idx):
        target_sizes = batch.pop("target_sizes")
        with torch.no_grad():
            outputs = self.forward(**batch)
        if "cls" in self.cfg.stage.name or self.cfg.stage.name == "test":
            cls_probs = torch.sigmoid(outputs.cls_logits).squeeze(-1).cpu().tolist()
        else:
            cls_probs = None
        if "seg" in self.cfg.stage.name or self.cfg.stage.name == "test":
            segmentation = self.processor.post_process_segmentation(
                outputs, target_sizes=target_sizes
            )
            pred_masks = [
                Image.fromarray(
                    torch.where(seg == 0, 255, 0).to(dtype=torch.uint8).cpu().numpy()
                )
                for seg in segmentation
            ]
            if (
                getattr(self.cfg.stage, "trust_classifier", None)
                and cls_probs is not None
            ):
                threshold = (
                    self.cfg.stage.trust_classifier
                )  # whether to trust the classifier
                for i, mask in enumerate(pred_masks):
                    # if cls prob < threshold, it means classifier thinks it is real
                    # so we use a zero mask
                    if cls_probs[i] < threshold:
                        pred_masks[i] = Image.new("L", mask.size, 0)

        return {"pred_masks": pred_masks, "cls_probs": cls_probs}
