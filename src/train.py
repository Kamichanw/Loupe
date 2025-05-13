import os
import re
import shutil
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
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict,
)
from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import DictConfig
from pathlib import Path
from safetensors.torch import save_file

from models.loupe import LoupeModel, LoupeConfig
from data_module import DataModule
from lit_model import LitModel

sys.path.insert(0, ".")
project_root = Path(__file__).resolve().parent.parent
save_path = os.environ.get("CATTINO_TASK_HOME", f"{project_root}/results")
checkpoint_callback: ModelCheckpoint


@rank_zero_only
def convert_deepspeed_checkpoint(cfg: DictConfig):
    """
    Convert deepspeed checkpoint to fp32 safetensors format.
    All frozen parameters will be removed.
    """
    converted_save_dir = os.path.join(project_root, "checkpoints", cfg.stage.name)
    if os.path.isdir(converted_save_dir) and os.listdir(converted_save_dir):
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
        converted_save_dir += f"_{max_num + 1}"
    os.makedirs(converted_save_dir, exist_ok=True)
    convert_zero_checkpoint_to_fp32_state_dict(
        checkpoint_callback.best_model_path,
        os.path.join(converted_save_dir, "fp32_state_dict.pth"),
    )
    ckpt = torch.load(
        os.path.join(converted_save_dir, "fp32_state_dict.pth"),
        map_location="cpu",
        weights_only=True,
    )
    for param in list(ckpt["state_dict"].keys()):
        if getattr(cfg.model, "freeze_backbone", False) and param.startswith("loupe.backbone"):
            ckpt["state_dict"].pop(param)
        if getattr(cfg.model, "freeze_cls", False) and param.startswith("loupe.classifier"):
            ckpt["state_dict"].pop(param)
        if getattr(cfg.model, "freeze_seg", False) and param.startswith("loupe.segmentor"):
            ckpt["state_dict"].pop(param)
    save_file(ckpt["state_dict"], os.path.join(converted_save_dir, "model.safetensors"))
    print(f"Model converted to FP32 and saved to {converted_save_dir}.")

    os.remove(
        os.path.join(converted_save_dir, "fp32_state_dict.pth")
    )
    shutil.rmtree(checkpoint_callback.best_model_path)


@hydra.main(
    config_path=str(project_root / "configs"),
    config_name="base",
    version_base=None,
)
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)
    global checkpoint_callback
    checkpoint_callback = hydra.utils.instantiate(
        cfg.ckpt.saver,
        dirpath=os.path.join(save_path, "checkpoints"),
    )
    logger = TensorBoardLogger(
        save_dir=save_path,
        name="logs",
        sub_dir=cfg.stage.name,
        default_hp_metric=False,
    )
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[LearningRateMonitor(), RichProgressBar(), checkpoint_callback],
        fast_dev_run=cfg.fast_dev_run,
        devices=(
            1
            if cfg.fast_dev_run
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
        val_check_interval=0.05,
        log_every_n_steps=2,
        accumulate_grad_batches=cfg.hparams.accumulate_grad_batches,
    )
    torch.set_float32_matmul_precision("medium")
    loupe_config = LoupeConfig(stage=cfg.stage.name, **cfg.model)
    loupe = LoupeModel(loupe_config)

    model = LitModel(cfg, loupe)
    data_module = DataModule(cfg, loupe_config)

    trainer.fit(model, data_module)

    if "deepspeed" in cfg.strategy and not cfg.fast_dev_run:
        convert_deepspeed_checkpoint(cfg)


if __name__ == "__main__":
    main()
