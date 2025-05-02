export CATTINO_HOME=./results

meow set shutdown-on-complete False
# meow create \
    python src/train.py \
    stage=cls 
    # --min-devices 4 --requires-memory-per-device 40000 --task-name "pe_lr=1e-5"
    # -- \