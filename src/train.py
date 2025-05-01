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


from models.loupe import LoupeModel
from data_module import DataModule
from lit_model import LitModel

sys.path.insert(0, ".")
project_root = Path(__file__).resolve().parent.parent


@hydra.main(
    config_path=str(project_root / "configs"),
    config_name="base.yaml",
    version_base=None,
)
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)
    save_path = os.environ.get("CATTINO_TASK_HOME", f"{project_root}/results")

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(save_path, "checkpoints"),
        filename="loupe-{epoch}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    logger = TensorBoardLogger(
        save_dir=save_path,
        name="logs",
        default_hp_metric=False,
    )
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[LearningRateMonitor(), RichProgressBar(), checkpoint_callback],
        # fast_dev_run=True,
        # devices=1,
        max_epochs=cfg.epoch,
        devices=len(
            os.environ.get(
                "CUDA_VISIBLE_DEVICES",
                ",".join(str(i) for i in range(torch.cuda.device_count())),
            ).split(",")
        ),
        strategy=cfg.strategy,
        precision=cfg.precision,
        gradient_clip_val=cfg.hparams.grad_clip_val,
        val_check_interval=0.25,
        log_every_n_steps=2,
        accumulate_grad_batches=cfg.hparams.accumulate_grad_batches,
    )
    torch.set_float32_matmul_precision("medium")
    loupe_config = hydra.utils.instantiate(cfg.model.config)
    loupe = LoupeModel(loupe_config)

    model = LitModel(cfg, loupe)
    data_module = DataModule(cfg, loupe_config)

    trainer.fit(
        model,
        data_module,
    )


if __name__ == "__main__":
    main()
