#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# 1. 路径
# ============================================================
MODEL=/share/project/wuhaiming/data/models/Llama-3.1-8B
TRAIN=/share/project/wuhaiming/spaces/scs/data/sft/random/random_01.jsonl
DEV=/share/project/wuhaiming/spaces/scs/data/dev/oasst2/oasst2_validation.jsonl

EXP_NAME=Llama-3.1-8B-SFT-Random-01
OUTPUT_ROOT=/share/project/wuhaiming/spaces/scs/output/adapters
OUT=${OUTPUT_ROOT}/${EXP_NAME}

LOG_ROOT=/share/project/wuhaiming/spaces/scs/output/logs
LOG_FILE=${LOG_ROOT}/${EXP_NAME}.log

mkdir -p "$OUT"
mkdir -p "$LOG_ROOT"

# ============================================================
# 2. 基础检查
# ============================================================

if [[ ! -d "$MODEL" ]]; then
    echo "[ERROR] Model directory does not exist: $MODEL"
    exit 1
fi

if [[ ! -f "$TRAIN" ]]; then
    echo "[ERROR] Training dataset does not exist: $TRAIN"
    exit 1
fi

if [[ ! -f "$DEV" ]]; then
    echo "[ERROR] Validation dataset does not exist: $DEV"
    exit 1
fi

TRAIN_SIZE=$(wc -l < "$TRAIN")
DEV_SIZE=$(wc -l < "$DEV")

echo "============================================================"
echo "Experiment:      $EXP_NAME"
echo "Model:           $MODEL"
echo "Training data:   $TRAIN"
echo "Training size:   $TRAIN_SIZE"
echo "Validation data: $DEV"
echo "Validation size: $DEV_SIZE"
echo "Output:          $OUT"
echo "Log:             $LOG_FILE"
echo "============================================================"

# ============================================================
# 3. 环境
# ============================================================

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NPROC_PER_NODE=4

# 避免 tokenizer 多进程产生无关警告
export TOKENIZERS_PARALLELISM=false

# NCCL 稳定性设置
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN

# Hugging Face / ModelScope 离线环境按需启用
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1

# ============================================================
# 4. 训练
# ============================================================

swift sft \
    --output_dir "$OUT" \
    \
    --model "$MODEL" \
    --model_type llama \
    --template llama3_2 \
    --check_model true \
    --torch_dtype bfloat16 \
    \
    --dataset "$TRAIN" \
    --val_dataset "$DEV" \
    --dataset_shuffle true \
    --train_dataloader_shuffle true \
    --val_dataset_shuffle false \
    --dataset_num_proc 16 \
    --load_from_cache_file false \
    \
    --loss_scale default \
    --add_non_thinking_prefix false \
    --disable_ignore_empty_think true \
    \
    --max_length 4096 \
    --truncation_strategy delete \
    --packing false \
    \
    --tuner_type lora \
    --target_modules all-linear \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.0 \
    --lora_bias none \
    \
    --learning_rate 1e-4 \
    --num_train_epochs 4 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    \
    --optim adamw_torch \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --adam_epsilon 1e-8 \
    --weight_decay 0.0 \
    --max_grad_norm 1.0 \
    \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.01 \
    \
    --gradient_checkpointing true \
    \
    --eval_strategy steps \
    --eval_on_start true \
    --eval_steps 200 \
    \
    --save_strategy steps \
    --save_steps 200 \
    --save_total_limit 5 \
    --save_only_model false \
    \
    --load_best_model_at_end true \
    --metric_for_best_model eval_loss \
    --greater_is_better false \
    \
    --logging_strategy steps \
    --logging_first_step true \
    --logging_steps 10 \
    --report_to tensorboard \
    \
    --seed 42 \
    --data_seed 42 \
    \
    2>&1 | tee "$LOG_FILE"