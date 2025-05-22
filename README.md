# Loupe

The 1st place solution of IJCAI 2025 challenge track 1: Forgery Image Detection and Localization. The top 5 final leaderboard is as follows:
| User | Overall Score |
|:---:|:---:|
| Loupe (ours) | **0.846** |
| Rank 2 | 0.8161 |
| Rank 3 | 0.8151 |
| Rank 4 | 0.815 |
| Rank 5 | 0.815 |


## Setup
### 1. Create environment
```bash
conda create -y -n loupe python=3.11
conda activate loupe
pip install -r requirements.txt
mkdir ./pretrained_weights/PE-Core-L14-336
```

### 2. Prepare pretrained weights
Download [Perception Encoder](https://github.com/facebookresearch/perception_models) following their original instructions, and place `PE-Core-L14-336.pt` at `./pretrained_weights/PE-Core-L14-336`.

### 3. Prepare datasets
Download the dataset to any location of your choice. Then, use the [`dataset_preprocess.ipynb`](./dataset_preprocess.ipynb) notebook to preprocess the dataset. This process converts the dataset into a directly loadable `DatasetDict` and saves it in `parquet` format.

After preprocessing, you will obtain a dataset with three splits: `train`, `valid`, and `test`. Each item in these splits has the following structure:

```python
{
    "image": "path/to/image",  # but will be loaded as an actual PIL.Image.Image object
    "mask": "path/to/mask",    # set to None for real images without masks
    "name": "basename_of_image.png"
}
```

> [!NOTE]
> You can also adapt to a new dataset. In this case, you need to modify [`dataset_preprocess.ipynb`](./dataset_preprocess.ipynb) for your own use.

After preparation, the last thing to do is specifying `path/to/your/dataset` in [dataset.yaml](configs/dataset.yaml).

## How to train
Loupe employs a three-stage training process. The first stage trains the classifier and can be executed with the following command:

```bash
python src/train.py stage=cls
```

During training, two directories will be automatically created:

* `./results/checkpoints` — contains the DeepSpeed-format checkpoint with the highest AUC on the validation set (when using the default training strategy, which can be configured in `./configs/base.yaml`).
* `./results/logs` — contains logs in TensorBoard format. You can monitor the training progress by running:

```bash
tensorboard --logdir=./results/logs
```

After training completes, the best checkpoint will be saved in the directory `./checkpoints/cls-auc=xxx.ckpt`. This directory contains a single file `model.safetensors`, which stores the best checkpoint in the safetensors format.

The second stage trains the segmentation head. To do so, simply replace the command line argument `stage=cls` with `stage=seg` in the stage 1 command.

The third stage jointly trains the backbone, classifier head, and segmentation head. By default, a portion of the validation set is used as training data, while the remainder is reserved for validation. The reason why I use validation set as an extra training set is the test set used in the competition is slightly out-of-distribution (OOD). I found that 
if continue training on the original training set will result in overfitting.

All training configurations can be adjusted within the `configs/` directory. Detailed comments are provided to facilitate quick and clear configuration.

## How to test or predict
By default, testing is performed on the full validation set. This means it is not suitable for evaluating Loupe trained in the third stage, since the third stage trains Loupe on the validation set itself (see above). Alternatively, if you are willing to make a slight modification to [data loading process](./src/data_module.py) to have Loupe train on the training set instead, this limitation can be avoided. 

To evaluate a trained model, you can run:
```bash
python src/infer.py stage=test ckpt.checkpoint_paths=["checkpoints/cls/model.safetensors","checkpoints/seg/model.safetensors"]
```

The `ckpt.checkpoint_paths` configuration is defined under `configs/ckpt`. It is a list that specifies the checkpoints to load sequentially during execution.

The prediction step is essentially the same as the test step. You only need to add an additional parameter to specify the output directory for predictions. For example:

```bash
python src/infer.py stage=test \
    ckpt.checkpoint_paths=["checkpoints/cls/model.safetensors","checkpoints/seg/model.safetensors"] \
    stage.pred_output_dir=./pred_outputs
```

The classification predictions will be saved in `./pred_outputs/predictions.txt`, and the mask outputs will be stored in `./pred_outputs/masks`. For more details on available parameters, please refer to `configs/stage/test.yaml`.
