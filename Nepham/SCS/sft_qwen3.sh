MODEL=/share/project/wuhaiming/data/models/Qwen3-8B-Base
TRAIN=/share/project/wuhaiming/spaces/scs/data/sft/random_01.jsonl
DEV=/share/project/wuhaiming/spaces/CLLM/data/sft/dev/no_robots_test.jsonl
OUT=/share/project/wuhaiming/spaces/scs/output/models/Qwen3-8B-Base-SFT-Random-01


CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4  \
swift sft \
    --output_dir "$OUT" \
    --model "$MODEL" \
    --dataset "$TRAIN" \
    --check_model true \
    --enable_thinking false \
    --add_non_thinking_prefix true \
    --loss_scale last_round+ignore_empty_think \
    --max_length 4096 \
    --dataloader_num_workers 64 \
    --dataset_num_proc 64 \
    --load_from_cache_file true \
    --truncation_strategy delete \ \
    --dataset_shuffle true \
    --train_dataloader_shuffle true \
    --val_dataset_shuffle false \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --torch_dtype bfloat16 \
    --tuner_type lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --target_modules all-linear \
    --lr_scheduler_type linear \
    --learning_rate 8e-5 \
    --deepspeed zero2 \
    --num_train_epochs 10 \
    --optim adamw_torch \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --adam_epsilon 1e-8 \
    --weight_decay 0.0 \
    --warmup_ratio 0.05 \
    --gradient_checkpointing true \
    --load_best_model_at_end true \
    --metric_for_best_model eval_loss \
    --greater_is_better false \
    --eval_strategy steps \
    --save_strategy steps \
    --seed 42 \
    --data_seed 42 \
    --eval_steps 500 \
    --save_steps 500 \
    --save_total_limit 5 \
    --logging_steps 10

    # --model ../../models/Qwen3.5-9B \
    # --model ../ADC/charize/Qwen3.5-9B-Base-Char \
    # --custom_dataset_info Nepham/dataset_info.json \
    # --group_by_length true \
    # --lr_scheduler_type constant \
    # --tokenizer /share/project/wuhaiming/data/models/Qwen3.5-9B-Base \
    # --attn_impl flash_attention_2 \
    # --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    # --system Nepham/system_prompt/csc.txt \
