nproc_per_node=6

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=$nproc_per_node \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
swift sft \
    --output_dir Nepham/output/Adapter/Qwen35 \
    --model /share/project/wuhaiming/spaces/ms-swift/Nepham/output/Adapter/Qwen35/extract_from_v17 \
    --model_type qwen \
    --check_model false \
    --custom_dataset_info Nepham/dataset_info.json \
    --dataset 34m_confuse_gen \
    --system Nepham/system_prompt/csc.txt \
    --enable_thinking false \
    --tuner_type full \
    --deepspeed zero3 \
    --torch_dtype float16 \
    --dataloader_num_workers 4 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --learning_rate 7e-5 \
    --gradient_accumulation_steps 8 \
    --eval_steps 500 \
    --save_steps 500 \
    --save_total_limit 5 \
    --logging_steps 10 \
    --max_length 2048 \
    --warmup_ratio 0.05 \
    --load_from_cache_file true
    # --lr_scheduler_type constant \
    # --tokenizer /share/project/wuhaiming/data/models/Qwen3.5-9B-Base \
    # --attn_impl flash_attention_2 \
    # --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
