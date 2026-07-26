swift export \
    --adapters /share/project/wuhaiming/spaces/CLLM/adapters/char-cpt/v9-20260722-020006/checkpoint-30000 \
    --output_dir /share/project/wuhaiming/spaces/CLLM/models/Qwen3-4B-Base-Char-PT \
    --merge_lora true \
    --max_shard_size 5GB
