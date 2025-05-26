import os
import subprocess
import sys
import tempfile
import hydra
import torch
import pytorch_lightning as pl

from pytorch_lightning.callbacks import RichProgressBar, BasePredictionWriter
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import DictConfig
from pathlib import Path

from models.loupe import LoupeModel, LoupeConfig
from data_module import DataModule
from lit_model import LitModel

sys.path.insert(0, ".")
project_root = Path(__file__).resolve().parent.parent
save_path = os.environ.get("CATTINO_TASK_HOME", f"{project_root}/results")


@rank_zero_only
def prepare_output_dir(pred_path, mask_dir):
    if os.path.isfile(pred_path):
        os.remove(pred_path)
    if os.path.isdir(mask_dir):
        print(f"Removing existing directory: {mask_dir}...")
        try:
            with tempfile.TemporaryDirectory() as empty_dir:
                result = subprocess.run(
                    ["rsync", "-a", "--delete", empty_dir + "/", mask_dir + "/"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                if result.returncode != 0:
                    raise RuntimeError(f"rsync failed: {result.stderr}")
        except (FileNotFoundError, RuntimeError) as e:
            print(
                f"rsync not available or failed ({e}), overwriting previous results..."
            )
    os.makedirs(mask_dir, exist_ok=True)


class CustomWriter(BasePredictionWriter):
    def __init__(self, cfg: DictConfig, write_interval):
        super().__init__(write_interval)
        output_dir = cfg.stage.pred_output_dir
        self.mask_dir = os.path.join(output_dir, "masks")
        self.pred_path = os.path.join(output_dir, "predictions.txt")

        prepare_output_dir(self.pred_path, self.mask_dir)

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        cls_probs, pred_masks = prediction["cls_probs"], prediction["pred_masks"]
        with open(self.pred_path, "a") as f:
            for name, cls_prob in zip(batch["name"], cls_probs):
                f.write(f"{name},{cls_prob:.4f}\n")

        for name, pred_mask in zip(batch["name"], pred_masks):
            pred_mask.save(os.path.join(self.mask_dir, name))


sys.path.insert(0, ".")
project_root = Path(__file__).resolve().parent.parent


@hydra.main(
    config_path=str(project_root / "configs"),
    config_name="base",
    version_base=None,
)
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)

    if cfg.stage.name != "test":
        raise ValueError("This script is for testing only. Please use the test stage.")

    if cfg.stage.enable_tta:
        # if tta is enbaled, turn trainer to trainable
        trainer_overrides = dict(
            logger=TensorBoardLogger(
                save_dir=save_path,
                name="logs",
                sub_dir=cfg.stage.name,
                default_hp_metric=False,
            ),
            max_epochs=1,
            strategy=cfg.strategy,
            gradient_clip_val=cfg.hparams.grad_clip_val,
            val_check_interval=0.2,
            log_every_n_steps=2,
            accumulate_grad_batches=cfg.hparams.accumulate_grad_batches,
        )
        cfg.model.freeze_seg = False
        cfg.model.enable_conditional_queries = True
    else:
        trainer_overrides = dict(logger=False)

    callbacks = [RichProgressBar()]
    if cfg.stage.pred_output_dir:
        callbacks.append(CustomWriter(cfg=cfg, write_interval="batch"))

    trainer = pl.Trainer(
        callbacks=callbacks,
        fast_dev_run=cfg.fast_dev_run,
        devices=1 if cfg.fast_dev_run else "auto",
        precision=cfg.precision,
        enable_checkpointing=False,
        **trainer_overrides,
    )
    torch.set_float32_matmul_precision("medium")
    loupe_config = LoupeConfig(stage=cfg.stage.name, **cfg.model)
    loupe = LoupeModel(loupe_config)
    model = LitModel(cfg=cfg, loupe=loupe)
    data_module = DataModule(cfg, loupe_config)

    if cfg.stage.enable_tta:
        trainer.fit(
            model,
            data_module,
        )
    elif cfg.stage.pred_output_dir:
        trainer.predict(
            model,
            data_module,
            return_predictions=False,
        )
    else:
        trainer.test(
            model,
            data_module,
        )


if __name__ == "__main__":
    main()
