import os
import subprocess
import time
import sys

ckpt_dir = "/gemini/code/loupe/checkpoints"
while True:
    cls_seg_ckpt = [d for d in os.listdir(ckpt_dir) if d.startswith("cls_seg")]
    if cls_seg_ckpt:
        cls_seg_ckpt = cls_seg_ckpt[0]
        if os.path.exists(os.path.join(ckpt_dir, cls_seg_ckpt)):
            print(f"Found checkpoint: {cls_seg_ckpt}")
            ckpt_path = os.path.join(ckpt_dir, cls_seg_ckpt, 'model.safetensors')
            while not os.path.exists(ckpt_path):
                time.sleep(5)
            result = subprocess.run(
                [
                    sys.executable,
                    "src/infer.py",
                    "stage=test",
                    "stage.pred_output_dir=./pred_outputs",
                    f"ckpt.checkpoint_paths=[\"{ckpt_path}\"]",
                ],
            )
            if result.returncode != 0:
                print(f"Error running infer.py with checkpoint {ckpt_path}")
            else:
                subprocess.run(
                    [
                        "tar",
                        "-cf",
                        "pred_outputs.tar",
                        "pred_outputs"
                    ],
                    check=True
                )
            break
    else:
        time.sleep(10)
