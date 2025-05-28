import os
import shutil
import subprocess
import tempfile
import hydra
import torch

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, BasePredictionWriter
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict,
)
from safetensors.torch import save_file


@rank_zero_only
def convert_deepspeed_checkpoint(
    cfg: DictConfig, checkpoint_callback: ModelCheckpoint, output_dir: str
):
    """
    Convert deepspeed checkpoint to fp32 safetensors format.
    All frozen parameters will be removed.
    """
    os.makedirs(output_dir, exist_ok=True)
    convert_zero_checkpoint_to_fp32_state_dict(
        checkpoint_callback.best_model_path,
        os.path.join(output_dir, "fp32_state_dict.pth"),
    )
    with torch.serialization.safe_globals([set]):
        ckpt = torch.load(
            os.path.join(output_dir, "fp32_state_dict.pth"),
            map_location="cpu",
            weights_only=True,
        )
    for param in list(ckpt["state_dict"].keys()):
        if getattr(cfg.model, "freeze_backbone", False) and param.startswith(
            "loupe.backbone"
        ):
            ckpt["state_dict"].pop(param)
        if getattr(cfg.model, "freeze_cls", False) and param.startswith(
            "loupe.classifier"
        ):
            ckpt["state_dict"].pop(param)
        if getattr(cfg.model, "freeze_seg", False) and param.startswith(
            "loupe.segmentor"
        ):
            ckpt["state_dict"].pop(param)
    save_file(ckpt["state_dict"], os.path.join(output_dir, "model.safetensors"))
    OmegaConf.save(config=cfg, f=os.path.join(output_dir, "config.yaml"))
    OmegaConf.save(
        config=hydra.core.hydra_config.HydraConfig.get().overrides.task,
        f=os.path.join(output_dir, "overrides.yaml"),
    )
    print(f"Model converted to FP32 and saved to {output_dir}.")

    os.remove(os.path.join(output_dir, "fp32_state_dict.pth"))
    shutil.rmtree(checkpoint_callback.best_model_path)


@rank_zero_only
def prepare_output_dir(pred_path, mask_dir):
    if os.path.isfile(pred_path):
        os.remove(pred_path)
    if os.path.isdir(mask_dir):
        print(f"Removing existing directory: {mask_dir}...")
        try:
            with tempfile.TemporaryDirectory() as empty_dir:
                result = subprocess.run(
                    ["rsync", "-a", "--delete", empty_dir + "/", mask_dir + "/"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                if result.returncode != 0:
                    raise RuntimeError(f"rsync failed: {result.stderr}")
        except (FileNotFoundError, RuntimeError) as e:
            print(
                f"rsync not available or failed ({e}), overwriting previous results..."
            )
    os.makedirs(mask_dir, exist_ok=True)


class CustomWriter(BasePredictionWriter):
    def __init__(self, cfg: DictConfig, write_interval):
        super().__init__(write_interval)
        output_dir = cfg.stage.pred_output_dir
        self.mask_dir = os.path.join(output_dir, "masks")
        self.pred_path = os.path.join(output_dir, "predictions.txt")

        prepare_output_dir(self.pred_path, self.mask_dir)

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        cls_probs, pred_masks = prediction["cls_probs"], prediction["pred_masks"]
        with open(self.pred_path, "a") as f:
            for name, cls_prob in zip(batch["name"], cls_probs):
                f.write(f"{name},{cls_prob:.4f}\n")

        for name, pred_mask in zip(batch["name"], pred_masks):
            pred_mask.save(os.path.join(self.mask_dir, name))
