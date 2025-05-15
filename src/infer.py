import os
import sys
import hydra
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import DictConfig
from pathlib import Path


from models.loupe import LoupeModel, LoupeConfig
from data_module import DataModule
from lit_model import LitModel
from lightning.pytorch.callbacks import BasePredictionWriter


class CustomWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank
        torch.save(predictions, os.path.join(self.output_dir, f"predictions_{trainer.global_rank}.pt"))

        # optionally, you can also save `batch_indices` to get the information about the data index
        # from your prediction data
        torch.save(batch_indices, os.path.join(self.output_dir, f"batch_indices_{trainer.global_rank}.pt"))

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

    if cfg.stage.name not in ["test", "pred"]:
        raise ValueError(
            "This script is for testing only. Please use the test or pred stage."
        )

    logger = TensorBoardLogger(
        save_dir=save_path,
        name="logs",
        default_hp_metric=False,
    )
    callbacks = [RichProgressBar()]
    if cfg.stage.name == "pred":
        if not cfg.stage.pred_output_dir:
            raise ValueError("pred_output_dir must be set in pred stage")
        callbacks.append(
            CustomWriter(
                output_dir=cfg.stage.pred_output_dir,
                write_interval="epoch",
            )
        )
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=2,
        precision=cfg.precision,
        use_distributed_sampler=cfg.stage.name == "test",
    )
    torch.set_float32_matmul_precision("medium")
    loupe_config = LoupeConfig(stage=cfg.stage.name, **cfg.model)
    loupe = LoupeModel(loupe_config)
    model = LitModel(
        cfg=cfg,
        loupe=loupe,
    )
    data_module = DataModule(cfg, loupe_config)
    if cfg.stage.name == "test":
        trainer.test(
            model,
            data_module,
        )
    else:
        trainer.predict(
            model,
            data_module,
            return_predictions=False,
        )


if __name__ == "__main__":
    main()
