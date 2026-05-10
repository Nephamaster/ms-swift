nproc_per_node=4

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=$nproc_per_node \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
swift sft \
    --output_dir Nepham/output/Adapter/Qwen35-Char/ \
    --model ../ADC/charize/Qwen3.5-9B-Base-Char \
    --check_model false \
    --custom_dataset_info Nepham/dataset_info.json \
    --dataset 34m_mix \
    --system Nepham/system_prompt/csc.txt \
    --dataloader_num_workers 16 \
    --dataset_num_proc 16 \
    --load_from_cache_file true \
    --enable_thinking false \
    --add_non_thinking_prefix true \
    --loss_scale ignore_empty_think \
    --torch_dtype bfloat16 \
    --tuner_type lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --deepspeed zero2 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --eval_steps 500 \
    --save_steps 500 \
    --save_total_limit 5 \
    --logging_steps 10 \
    --max_length 2048 \
    --warmup_ratio 0.05 \
    
    # --model ../../data/models/Qwen3.5-9B-Base \
    # --model ../ADC/charize/Qwen3.5-9B-Base-Char \
    # --group_by_length true \
    # --lr_scheduler_type constant \
    # --tokenizer /share/project/wuhaiming/data/models/Qwen3.5-9B-Base \
    # --attn_impl flash_attention_2 \
    # --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
