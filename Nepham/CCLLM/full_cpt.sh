set -euo pipefail

MODEL=/share/project/wuhaiming/spaces/CC-LLM/models/Qwen3-1.7B-Base-Char-PGCA-ALIGN/best
VAL=/share/project/wuhaiming/spaces/CC-LLM/data/full_cpt/validation/validation.jsonl
OUT=/share/project/wuhaiming/spaces/CC-LLM/models/Qwen3-1.7B-Base-Char-PGCA-ALIGN-CPT/
TRAIN_DIR=/share/project/wuhaiming/spaces/CC-LLM/data/full_cpt/final
TRAIN_FILES=("$TRAIN_DIR"/train-{00000..00017}.jsonl)

for file in "${TRAIN_FILES[@]}"; do
    [[ -f "$file" ]] || {
        echo "Missing dataset shard: $file" >&2
        exit 1
    }
done

CUDA_VISIBLE_DEVICES=4,5,6,7 \
NPROC_PER_NODE=4  \
swift pt \
    --model "$MODEL" \
    --model_type qwen3 \
    --template qwen3 \
    --check_model false \
    --dataset "${TRAIN_FILES[@]}" \
    --val_dataset "$VAL" \
    --truncation_strategy split \
    --dataset_num_proc 32 \
    --load_from_cache_file false \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --learning_rate 1e-5 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_length 2048 \
    --packing false \
    --optim adamw_torch_fused \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-8 \
    --weight_decay 0.0 \
    --lr_scheduler_type linear \
    --max_grad_norm 1.0 \
    --warmup_ratio 0.03 \
    --logging_steps 10 \
    --gradient_checkpointing true \
    --eval_strategy steps \
    --save_strategy steps \
    --eval_steps 500 \
    --save_steps 500 \
    --save_total_limit 3 \
    --load_best_model_at_end true \
    --metric_for_best_model eval_loss \
    --greater_is_better false \
    --seed 42 \
    --data_seed 42 \
    --output_dir "$OUT"


# truncation_strategy split 只适用于预训练
