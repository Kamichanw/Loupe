from functools import partial
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
        images = self.processor.preprocess([x["image"].convert("RGB") for x in batch])
        masks = self.processor.preprocess_masks(
            [x["mask"].convert("L") if x["mask"] is not None else None for x in batch]
        )
        class_labels = [
            x["mask"] is not None for x in batch
        ]  # mask is None means it is real

        return {
            "pixel_values": torch.stack(images["pixel_values"]),  # (N, C, H, W)
            "mask_labels": torch.stack(masks["pixel_values"]).squeeze(
                1
            ),  # (N, 1, H, W) -> (N, H, W)
            "class_labels": torch.tensor(class_labels, dtype=torch.long),  # (N,)
            "patch_labels": (
                torch.stack(masks["patch_labels"])  # (N, num_patches, num_patches)
                if self.model_config.enable_patch_cls
                else None
            ),
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
        image_list = [x["image"].convert("RGB") for x in batch]
        mask_list = [
            x["mask"].convert("L") if x["mask"] is not None else None for x in batch
        ]

        if maybe_no_labels and all(x["image"] is None for x in batch):
            # if there are no labels, we don't convert images to a batched tensor
            images = self.processor.preprocess(image_list, do_resize=False)
            pixel_values = images["pixel_values"]
            mask_labels, masks, class_labels = None, None, None
        else:
            images = self.processor.preprocess(image_list)
            pixel_values = torch.stack(images["pixel_values"])
            masks = self.processor.preprocess_masks(
                mask_list, return_patch_labels=False
            )
            # (N, 1, H, W) -> (N, H, W)
            mask_labels = torch.stack(masks["pixel_values"]).squeeze(1)

            raw_masks = self.processor.preprocess_masks(
                mask_list, return_patch_labels=False, do_resize=False
            )
            # (N, H_i, W_i)
            masks = [mask.squeeze(0) for mask in raw_masks["pixel_values"]]
            for i, mask in enumerate(masks):
                if mask_list[i] is None:
                    # note that in PIL image, the size is (W, H)
                    masks[i] = torch.zeros(
                        (image_list[i].size[1], image_list[i].size[0]),
                    )

            # (N,)
            class_labels = torch.tensor(
                [x["mask"] is not None for x in batch],
                dtype=torch.long,
            )

        return {
            "pixel_values": pixel_values,  # (N, C, H, W) or (N, C, H_i, W_i)
            "mask_labels": mask_labels,  # (N, H, W) or None
            "class_labels": class_labels,  # (N,) or None
            "masks": masks,  # (N, H_i, W_i) or None
        }

    def test_dataloader(self):
        return DataLoader(
            self.testset,
            batch_size=self.cfg.hparams.batch_size,
            num_workers=self.cfg.num_workers,
            collate_fn=partial(self.test_collate_fn, maybe_no_labels=True),
            shuffle=False,
        )
