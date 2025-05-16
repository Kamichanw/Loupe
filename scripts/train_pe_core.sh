export CATTINO_HOME=./results

meow set shutdown-on-complete False
meow set override-exist-tasks allow
meow create \
    "python src/train.py" \
    --min-devices 4 --requires-memory-per-device 40000 --task-name "seg-alpha=0.3" \
    -- \
    stage=seg \
    model.tversky_alpha=0.3