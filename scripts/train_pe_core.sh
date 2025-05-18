export CATTINO_HOME=./results

meow set shutdown-on-complete False
meow set override-exist-tasks allow
meow create \
    "python src/infer.py" \
    --min-devices 4 --requires-memory-per-device 40000 --task-name "pred" \
    -- \
    stage=test \
    stage.pred_output_dir=/gemini/code/loupe/pred_outputs