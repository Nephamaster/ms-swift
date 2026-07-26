MODEL=/share/project/wuhaiming/data/models/Llama-3.1-8B
DATA=/tmp/debug64.jsonl
OUT=/share/project/wuhaiming/spaces/scs/output/debug/llama31-overfit64

CUDA_VISIBLE_DEVICES=0 \
NPROC_PER_NODE=1 \
swift sft \
    --output_dir "$OUT" \
    --model "$MODEL" \
    --dataset "$DATA" \
    --val_dataset "$DATA" \
    --model_type llama \
    --template llama3_2 \
    --loss_scale default \
    --add_non_thinking_prefix false \
    --tuner_type lora \
    --torch_dtype bfloat16 \
    --target_modules all-linear \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.0 \
    --learning_rate 1e-4 \
    --max_steps 200 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --max_length 4096 \
    --truncation_strategy delete \
    --optim adamw_torch \
    --lr_scheduler_type constant \
    --warmup_ratio 0 \
    --gradient_checkpointing true \
    --dataset_shuffle false \
    --train_dataloader_shuffle false \
    --eval_strategy steps \
    --eval_on_start true \
    --eval_steps 20 \
    --save_strategy no \
    --logging_steps 1 \
    --seed 42 \
    --data_seed 42
