nproc_per_node=4

NPROC_PER_NODE=$nproc_per_node \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift pt \
    --output_dir Nepham/output/ADC/Qwen3-4B-Base-Char-PT \
    --model ../ADC/charize/Qwen3-4B-Base-Char \
    --model_type qwen3 \
    --template qwen3 \
    --check_model false \
    --custom_dataset_info Nepham/dataset_info.json \
    --dataset tiger_pt_zh \
    --tuner_type lora \
    --target_modules all-linear \
    --modules_to_save embed_tokens lm_head \
    --lora_rank 16 \
    --lora_alpha 32 \
    --torch_dtype bfloat16 \
    --streaming true \
    --max_steps 88000 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps $(expr 64 / $nproc_per_node) \
    --eval_steps 500 \
    --save_steps 500 \
    --save_total_limit 5 \
    --logging_steps 10 \
    --deepspeed zero3 \
    --max_length 4096 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --save_only_model true

    #
    # --packing true \
    # --attn_impl flash_attn
