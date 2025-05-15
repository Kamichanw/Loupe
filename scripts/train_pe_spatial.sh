export CATTINO_HOME=./results

meow set shutdown-on-complete False
# meow create \
    python src/train.py \
    stage=cls \
    hparams.backbone_lr=null \
    hparams.batch_size=32 \
    hparams.accumulate_grad_batches=2 \
    model.freeze_backbone=True \
    model.backbone_path=pretrained_weights/pe-spatial/PE-Spatial-G14-448.pt \
    model.backbone_name=PE-Spatial-G14-448 \
    +model.backbone_overrides.pool_type='tok' \
    +model.use_cls_token=True 
    # --min-devices 4 --requires-memory-per-device 40000 --task-name "pe_lr=1e-5"
    # -- \