import os
import shutil
import subprocess
import sys
import tempfile
import hydra
import torch
import pytorch_lightning as pl

from pytorch_lightning.callbacks import RichProgressBar, BasePredictionWriter
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from omegaconf import DictConfig
from pathlib import Path

from models.loupe import LoupeModel, LoupeConfig
from data_module import DataModule
from lit_model import LitModel


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

    callbacks = [RichProgressBar()]
    if cfg.stage.pred_output_dir:
        callbacks.append(CustomWriter(cfg=cfg, write_interval="batch"))
    trainer = pl.Trainer(
        logger=False,
        callbacks=callbacks,
        precision=cfg.precision,
        fast_dev_run=cfg.fast_dev_run,
        enable_checkpointing=False,
    )
    torch.set_float32_matmul_precision("medium")
    loupe_config = LoupeConfig(stage=cfg.stage.name, **cfg.model)
    loupe = LoupeModel(loupe_config)
    model = LitModel(cfg=cfg, loupe=loupe)
    model.eval()
    data_module = DataModule(cfg, loupe_config)
    if cfg.stage.pred_output_dir:
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
