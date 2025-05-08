from functools import partial
import numpy as np
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
        if stage in [None, "validate", "fit"]:
            self.trainset = dataset["train"]
            # use a small subset to prevent too long validation time
            self.validset = (
                dataset["validation"].shuffle(seed=self.cfg.seed).select(range(5000))
            )
        else:
            self.testset = dataset["validation"]

    def train_collate_fn(self, batch):
        images = [x["image"] for x in batch]
        masks = [x["mask"] for x in batch]
        labels = [x is not None for x in masks]  # mask is None means it is real

        return {
            **self.processor(
                images, masks, self.model_config.enable_patch_cls, return_tensors="pt"
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

    def test_collate_fn(self, batch, maybe_no_labels: bool = False):
        """
        Collate function for valid and test dataloaders.
        Args:
            batch: List of dictionaries containing "image" and "mask" keys.
            maybe_no_labels: If True, check if all images are None. If so, set mask_labels, masks, and class_labels to None.
                Only set to True when using test_dataloader.
        """
        images = [x["image"] for x in batch]
        masks = [x["mask"] for x in batch]
        labels = [x is not None for x in masks]  # mask is None means it is real

        if maybe_no_labels and not any(labels):
            outputs = self.processor(images, return_tensors="pt")
            labels, masks = None, None
        else:
            outputs = self.processor(images, masks, return_tensors="pt")
            for i, mask in enumerate(masks):
                ignore_value = self.model_config.mask2former_config.ignore_value
                if mask is None:
                    # note that in PIL image, the size is (W, H)
                    masks[i] = torch.full(
                        (images[i].size[1], images[i].size[0]),
                        ignore_value,
                    )
                else:
                    # convert to binary mask with 0 and 1
                    mask = self.processor.convert_to_binary_masks(mask)
                    # reduce labels
                    masks[i] = torch.where(mask == 0, ignore_value, mask - 1)

        return {
            **outputs,
            "masks": masks,  # a list of (N, H_i, W_i) or None
            "labels": labels,  # (N,) or None
        }

    def test_dataloader(self):
        return DataLoader(
            self.testset,
            batch_size=self.cfg.hparams.batch_size,
            num_workers=self.cfg.num_workers,
            collate_fn=partial(self.test_collate_fn, maybe_no_labels=True),
            shuffle=False,
        )
