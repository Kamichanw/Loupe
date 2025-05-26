from functools import partial
import os
import numpy as np
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from datasets import load_dataset, concatenate_datasets
from models.loupe import LoupeImageProcessor, LoupeConfig


class DataModule(pl.LightningDataModule):

    def __init__(self, cfg: DictConfig, model_config: LoupeConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.model_config = model_config
        self.processor = LoupeImageProcessor(self.model_config)

    def setup(self, stage: str) -> None:
        dataset = load_dataset("parquet", data_dir=self.cfg.data_dir)
        if stage in [None, "validate", "fit"]:
            validset = dataset["validation"]
            if isinstance(self.cfg.valid_size, int):
                assert 0 < self.cfg.valid_size < len(validset)
                valid_size = self.cfg.valid_size
            elif isinstance(self.cfg.valid_size, float):
                assert 0 < self.cfg.valid_size <= 1
                valid_size = int(self.cfg.valid_size * len(validset))
            else:
                raise ValueError(
                    f"Invalid valid_size: {self.cfg.valid_size}. It should be either int or float."
                )

            # use a small subset to prevent too long validation time
            additional_trainset, validset = validset.train_test_split(
                test_size=valid_size, seed=self.cfg.seed, shuffle=True
            ).values()
            self.validset = validset

            # for the 3th stage training, we only use the additional trainset splitted from the validation set
            if self.cfg.stage.name in ["cls_seg", "test"]:
                self.trainset = additional_trainset
            else:
                self.trainset = dataset["train"]
        elif stage == "test":
            self.testset = dataset["validation"]
        elif stage == "predict":
            self.testset = dataset["test"]

    def train_collate_fn(self, batch):
        images = [x["image"] for x in batch]
        masks = [x["mask"] for x in batch]
        labels = [x is not None for x in masks]  # mask is None means it is real

        return {
            **self.processor(
                images,
                masks if not getattr(self.cfg.stage, "enable_tta", False) else None,
                self.model_config.enable_patch_cls,
                return_tensors="pt",
            ),
            "labels": torch.tensor(labels, dtype=torch.long),  # (N,)
        }

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.cfg.hparams.batch_size,
            num_workers=self.cfg.num_workers,
            collate_fn=self.train_collate_fn,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validset,
            batch_size=self.cfg.hparams.batch_size,
            num_workers=self.cfg.num_workers,
            collate_fn=self.test_collate_fn,
            shuffle=False,
        )

    def test_collate_fn(self, batch):
        """
        Collate function for valid and test dataloaders.
        Args:
            batch: List of dictionaries containing "image" and "mask" keys.
        """
        images = [x["image"] for x in batch]
        masks = [x["mask"] for x in batch]
        labels = [x is not None for x in masks]  # mask is None means it is real

        outputs = self.processor(images, masks, return_tensors="pt")
        for i, mask in enumerate(masks):
            if mask is None:
                # note that in PIL image, the size is (W, H)
                masks[i] = torch.zeros(
                    (images[i].size[1], images[i].size[0]),
                    dtype=torch.uint8,
                )
            else:
                # convert to binary mask with 0 and 1
                masks[i] = self.processor.convert_to_binary_masks(mask)

        return {
            **outputs,
            "masks": masks,  # a list of (N, H_i, W_i)
            "labels": (torch.tensor(labels, dtype=torch.long)),  # (N,)
        }

    def test_dataloader(self):
        return DataLoader(
            self.testset,
            batch_size=self.cfg.hparams.batch_size,
            num_workers=self.cfg.num_workers,
            collate_fn=self.test_collate_fn,
        )

    def predict_collate_fn(self, batch):
        """
        Collate function for predict dataloader.
        Args:
            batch: List of dictionaries containing "image" and "mask" keys.
        """
        images = [x["image"] for x in batch]

        outputs = self.processor(images, return_tensors="pt")

        return {
            **outputs,
            "target_sizes": [image.size[::-1] for image in images],
            "name": [os.path.basename(x["path"]) for x in batch],
        }

    def predict_dataloader(self):
        return DataLoader(
            self.testset,
            batch_size=self.cfg.hparams.batch_size,
            num_workers=self.cfg.num_workers,
            collate_fn=self.predict_collate_fn,
            shuffle=False,
        )
