set -euo pipefail

MODEL=/share/project/wuhaiming/spaces/CC-LLM/models/Qwen3-1.7B-Base-Char-PGCA
VAL=/share/project/wuhaiming/spaces/CC-LLM/data/corpus/runs/phase1/phase1-1b-default-c97f9387391d/final/ms_swift/validation/validation-00000.jsonl
OUT=/share/project/wuhaiming/spaces/CC-LLM/models/Qwen3-1.7B-Base-Char-PGCA-ALIGN/
TRAIN_DIR=/share/project/wuhaiming/spaces/CC-LLM/data/corpus/runs/phase1/phase1-1b-default-c97f9387391d/final/ms_swift/train
TRAIN_FILES=("$TRAIN_DIR"/train-{00000..00005}.jsonl)

for file in "${TRAIN_FILES[@]}"; do
    [[ -f "$file" ]] || {
        echo "Missing dataset shard: $file" >&2
        exit 1
    }
done

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8  \
swift pt \
    --external_plugins CCLLM/diagnose_eval_nan.py \
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
    --freeze_parameters_ratio 0 \
    --freeze_parameters_regex '(^model\.layers\.[0-9]+\.pgca_attn\.|^model\.feature_memory_builder\.feature_embedding\.)' \
    --learning_rate 2e-5 \
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
    --eval_on_start true \
    --logging_nan_inf_filter false \
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
