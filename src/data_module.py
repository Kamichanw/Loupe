from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from datasets import load_dataset
from models.loupe import LoupeImageProcessor, LoupeConfig


class DataModule(pl.LightningDataModule):

    def __init__(self, cfg: DictConfig, model_config: LoupeConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.model_config = model_config
        self.processor = LoupeImageProcessor(self.model_config)

    def setup(self, stage: str) -> None:
        dataset = load_dataset("parquet", data_dir=self.cfg.data_dir)
        if stage == "fit" or stage is None:
            self.trainset = dataset["train"]
        elif stage == "validate":
            # use a small subset to prevent too long validation time
            self.validset = (
                dataset["validation"].shuffle(seed=self.cfg.seed).select(range(5000))
            )
        else:
            self.testset = dataset["validation"]

    def collate_fn(self, batch):
        images = self.processor.preprocess([x["image"].convert("RGB") for x in batch])
        masks = self.processor.preprocess_masks(
            [x["mask"].convert("L") if x["mask"] is not None else None for x in batch]
        )
        labels = [x["mask"] is not None for x in batch]  # mask is None means it is real

        return {
            "images": torch.stack(images["pixel_values"]),  # (N, C, H, W)
            "masks": torch.stack(masks["pixel_values"]),  # (N, H, W)
            "labels": torch.tensor(labels, dtype=torch.long),  # (N,)
            "patch_labels": (
                torch.stack(masks["patch_labels"])  # (N, num_patches, num_patches)
                if self.model_config.enable_patch_cls
                else None
            ),
        }

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            collate_fn=self.collate_fn,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            collate_fn=self.collate_fn,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.testset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            collate_fn=self.collate_fn,
            shuffle=False,
        )
