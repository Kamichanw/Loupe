import os
import sys
import hydra
import torch
import pytorch_lightning as pl

from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import DictConfig
from pathlib import Path

from models.loupe import LoupeModel, LoupeConfig
from data_module import DataModule
from lit_model import LitModel
from utils import CustomWriter, convert_deepspeed_checkpoint

sys.path.insert(0, ".")
project_root = Path(__file__).resolve().parent.parent
save_path = os.environ.get("CATTINO_TASK_HOME", f"{project_root}/results")


@hydra.main(
    config_path=str(project_root / "configs"),
    config_name="infer",
    version_base=None,
)
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)

    if cfg.stage.name != "test":
        raise ValueError("This script is for testing only. Please use the test stage.")

    callbacks = [RichProgressBar()]
    trainer_overrides = dict(
        devices=1 if cfg.trainer.fast_dev_run else "auto",
    )
    if cfg.stage.enable_tta:
        # if tta is enbaled, turn trainer to trainable
        trainer_overrides.update(
            logger=TensorBoardLogger(
                save_dir=save_path,
                name=cfg.stage.name,
                default_hp_metric=False,
            ),
            max_epochs=cfg.hparams.epoch,
            gradient_clip_val=cfg.hparams.grad_clip_val,
            val_check_interval=0.2,
            log_every_n_steps=2,
            accumulate_grad_batches=cfg.hparams.accumulate_grad_batches,
        )
        cfg.model.freeze_seg = False
        cfg.model.enable_conditional_queries = True
    else:
        trainer_overrides.update(
            logger=False,
            enable_checkpointing=False,
        )
        if cfg.stage.pred_output_dir:
            callbacks.append(CustomWriter(cfg=cfg, write_interval="batch"))
    trainer_overrides.update(cfg.trainer)

    if trainer_overrides.get("enable_checkpointing", False):
        checkpoint_callback = hydra.utils.instantiate(
            cfg.ckpt.saver,
            dirpath=os.path.join(save_path, "checkpoints"),
        )
        callbacks.append(checkpoint_callback)

    trainer = pl.Trainer(
        callbacks=callbacks,
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
