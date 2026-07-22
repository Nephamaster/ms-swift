swift export \
    --adapters /share/project/wuhaiming/spaces/CLLM/adapters/char-cpt-sft/v1-20260721-161626/checkpoint-10600 \
    --output_dir /share/project/wuhaiming/spaces/CLLM/models/Qwen3-4B-Base-Char-PT-SFT \
    --merge_lora true \
    --max_shard_size 5GB
