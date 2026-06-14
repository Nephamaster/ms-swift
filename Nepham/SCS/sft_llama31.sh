nproc_per_node=3

MASTER_PORT=29501 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=$nproc_per_node \
CUDA_VISIBLE_DEVICES=4,5,6 \
swift sft \
    --output_dir Nepham/output/SCS/Llama31-novel \
    --model ../../data/models/Llama-3.1-8B \
    --model_type llama \
    --template llama3_2 \
    --check_model true \
    --dataset Nepham/dataset/SFT_novel_10000.jsonl \
    --dataloader_num_workers 16 \
    --dataset_num_proc 16 \
    --load_from_cache_file true \
    --loss_scale last_round \
    --torch_dtype bfloat16 \
    --tuner_type lora \
    --lora_rank 16 \
    --lora_alpha 16 \
    --target_modules all-linear \
    --deepspeed zero2 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 8e-5 \
    --eval_steps 500 \
    --save_steps 500 \
    --save_total_limit 5 \
    --logging_steps 10 \
    --max_length 4096 \
    --warmup_ratio 0.01 \
    
    # --model ../../models/Qwen3.5-9B \
    # --model ../ADC/charize/Qwen3.5-9B-Base-Char \
    # --custom_dataset_info Nepham/dataset_info.json \
    # --group_by_length true \
    # --lr_scheduler_type constant \
    # --tokenizer /share/project/wuhaiming/data/models/Qwen3.5-9B-Base \
    # --attn_impl flash_attention_2 \
    # --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    # --system Nepham/system_prompt/csc.txt \
