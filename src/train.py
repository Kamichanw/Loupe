import os
import sys
import hydra
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    RichProgressBar,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import DictConfig
from pathlib import Path

from models.loupe import LoupeModel, LoupeConfig
from data_module import DataModule
from lit_model import LitModel
from utils import convert_deepspeed_checkpoint

sys.path.insert(0, ".")
project_root = Path(__file__).resolve().parent.parent
save_path = os.environ.get("CATTINO_TASK_HOME", f"{project_root}/results")


@hydra.main(
    config_path=str(project_root / "configs"),
    config_name="train",
    version_base=None,
)
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)

    if cfg.stage.name == "test":
        raise ValueError(
            "This script is for training only. Please use one of cls, seg, or cls_seg stages."
        )

    checkpoint_callback = hydra.utils.instantiate(
        cfg.ckpt.saver,
        dirpath=os.path.join(save_path, "checkpoints"),
    )
    logger = TensorBoardLogger(
        save_dir=save_path,
        name=cfg.stage.name,
        default_hp_metric=False,
    )
    trainer_overrides = dict(
        devices=1 if cfg.trainer.fast_dev_run else "auto",
        max_epochs=cfg.hparams.epoch,
        val_check_interval=0.05,
        log_every_n_steps=2,
        accumulate_grad_batches=cfg.hparams.accumulate_grad_batches,
        gradient_clip_val=cfg.hparams.grad_clip_val,
    )
    trainer_overrides.update(cfg.trainer)
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[LearningRateMonitor(), RichProgressBar(), checkpoint_callback],
        **trainer_overrides,
    )
    torch.set_float32_matmul_precision("medium")
    loupe_config = LoupeConfig(stage=cfg.stage.name, **cfg.model)
    loupe = LoupeModel(loupe_config)

    model = LitModel(cfg, loupe)
    data_module = DataModule(cfg, loupe_config)

    trainer.fit(model, data_module)

    if "deepspeed" in cfg.trainer.get("strategy", "") and trainer_overrides.get(
        "enable_checkpointing", False
    ):
        convert_deepspeed_checkpoint(
            cfg,
            checkpoint_callback,
            os.path.join(
                project_root,
                "checkpoints",
                os.path.basename(checkpoint_callback.best_model_path),
            ),
        )


if __name__ == "__main__":
    main()
