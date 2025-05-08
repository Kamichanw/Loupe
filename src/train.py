import os
import re
import subprocess
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
    fast_dev_run = False
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[LearningRateMonitor(), RichProgressBar(), checkpoint_callback],
        fast_dev_run=fast_dev_run,
        devices=(
            1
            if fast_dev_run
            else len(
                os.environ.get(
                    "CUDA_VISIBLE_DEVICES",
                    ",".join(str(i) for i in range(torch.cuda.device_count())),
                ).split(",")
            )
        ),
        max_epochs=cfg.hparams.epoch,
        strategy=cfg.strategy,
        precision=cfg.precision,
        gradient_clip_val=cfg.hparams.grad_clip_val,
        val_check_interval=0.25,
        log_every_n_steps=2,
        accumulate_grad_batches=cfg.hparams.accumulate_grad_batches,
    )
    torch.set_float32_matmul_precision("medium")
    loupe_config = LoupeConfig(stage=cfg.stage.name, **cfg.model)
    loupe = LoupeModel(loupe_config)

    model = LitModel(cfg, loupe)
    data_module = DataModule(cfg, loupe_config)

    trainer.fit(
        model,
        data_module,
    )

    if "deepspeed" in cfg.strategy and not fast_dev_run:
        converted_save_path = os.path.join(project_root, "checkpoints", cfg.stage.name)
        if os.path.isdir(converted_save_path) and os.listdir(converted_save_path):
            exist_dirs = [
                os.path.join(project_root, "checkpoints", d)
                for d in os.listdir(os.path.join(project_root, "checkpoints"))
                if os.path.isdir(os.path.join(project_root, "checkpoints", d))
                and d.startswith(cfg.stage.name)
            ]
            nums = re.findall(
                rf"{re.escape(cfg.stage.name)}_(\d+)",
                " ".join(os.path.basename(d) for d in exist_dirs),
            )
            max_num = max(map(int, nums)) if nums else 0
            converted_save_path += f"_{max_num + 1}"
        result = subprocess.run(
            [
                sys.executable,
                os.path.join(checkpoint_callback.best_model_path, "zero_to_fp32.py"),
                checkpoint_callback.best_model_path,
                converted_save_path,
                "--safe_serialization",
            ]
        )
        if result.returncode != 0:
            print("Error converting model to FP32.")
            sys.exit(1)
        else:
            print(f"Model converted to FP32 and saved to {converted_save_path}.")


if __name__ == "__main__":
    main()
