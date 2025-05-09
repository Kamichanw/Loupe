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

sys.path.insert(0, ".")
project_root = Path(__file__).resolve().parent.parent


@hydra.main(
    config_path=str(project_root / "configs"),
    config_name="base",
    version_base=None,
)
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)
    save_path = os.environ.get("CATTINO_TASK_HOME", f"{project_root}/results")

    if cfg.stage.name != "test":
        raise ValueError("This script is for testing only. Please use the test stage.")

    logger = TensorBoardLogger(
        save_dir=save_path,
        name="logs",
        default_hp_metric=False,
    )
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[RichProgressBar()],
        log_every_n_steps=2,
    )
    torch.set_float32_matmul_precision("medium")
    loupe_config = LoupeConfig(stage=cfg.stage.name, **cfg.model)
    loupe = LoupeModel(loupe_config)

    model = LitModel.load_from_checkpoint(
        cfg.checkpoint_path,
        strict=False,
        cfg=cfg,
        loupe=loupe,
    )
    data_module = DataModule(cfg, loupe_config)

    trainer.test(
        model,
        data_module,
    )


if __name__ == "__main__":
    main()
