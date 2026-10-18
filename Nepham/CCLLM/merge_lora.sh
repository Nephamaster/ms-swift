swift export \
    --adapters /share/project/wuhaiming/spaces/CC-LLM/adapters/csc-char-pgca-lora/v0-20261013-201902/checkpoint-10000 \
    --output_dir /share/project/wuhaiming/spaces/CC-LLM/models/Qwen3-1.7B-Base-Char-PGCA-CSC \
    --merge_lora true \
    --max_shard_size 5GB
