set -euo pipefail

MODEL=/share/project/wuhaiming/spaces/CLLM/models/Qwen3-4B-Base-Char
TRAIN=/share/project/wuhaiming/spaces/CLLM/data/processed/cpt/tiger_pretrain_zh_train.jsonl
DEV=/share/project/wuhaiming/spaces/CLLM/data/processed/cpt/tiger_pretrain_zh_dev.jsonl
OUT=/share/project/wuhaiming/spaces/CLLM/adapters/char-cpt

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8  \
swift pt \
    --model "$MODEL" \
    --model_type qwen3 \
    --template qwen3 \
    --check_model false \
    --dataset "$TRAIN" \
    --val_dataset "$DEV" \
    --dataset_num_proc 64 \
    --load_from_cache_file true \
    --tuner_type lora \
    --torch_dtype bfloat16 \
    --loss_scale default \
    --target_modules all-linear \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --learning_rate 1e-5 \
    --max_steps 30000 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --max_length 1024 \
    --packing false \
    --optim adamw_torch \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --adam_epsilon 1e-8 \
    --weight_decay 0.0 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.0 \
    --gradient_checkpointing true \
    --eval_strategy steps \
    --save_strategy steps \
    --eval_steps 500 \
    --save_steps 500 \
    --logging_steps 10 \
    --save_total_limit 3 \
    --load_best_model_at_end true \
    --metric_for_best_model eval_loss \
    --greater_is_better false \
    --seed 42 \
    --data_seed 42 \
    --output_dir "$OUT"
