#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# 固定配置
# ============================================================

MODEL=/share/project/wuhaiming/data/models/Llama-3.1-8B

DATA_ROOT=/share/project/wuhaiming/spaces/scs/data/sft/random
OUTPUT_ROOT=/share/project/wuhaiming/spaces/scs/output/adapters
LOG_ROOT=/share/project/wuhaiming/spaces/scs/output/logs/random

DEV=/share/project/wuhaiming/spaces/scs/data/dev/candidate_1000.jsonl

START_INDEX=1
END_INDEX=12

GPU_IDS=0,1
NPROC=2

mkdir -p "$OUTPUT_ROOT"
mkdir -p "$LOG_ROOT"

export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN

# ============================================================
# 基础检查
# ============================================================

if [[ ! -d "$MODEL" ]]; then
    echo "[ERROR] 模型目录不存在：$MODEL"
    exit 1
fi

if [[ ! -f "$DEV" ]]; then
    echo "[ERROR] 验证集不存在：$DEV"
    exit 1
fi

echo "============================================================"
echo "Model:       $MODEL"
echo "Data root:   $DATA_ROOT"
echo "Dev:         $DEV"
echo "Output root: $OUTPUT_ROOT"
echo "Range:       random_$(printf '%02d' "$START_INDEX")"
echo "             random_$(printf '%02d' "$END_INDEX")"
echo "GPU:         $GPU_IDS"
echo "============================================================"

# ============================================================
# 依次训练 random_01 ～ random_12
# ============================================================

for INDEX in $(seq "$START_INDEX" "$END_INDEX"); do
    ID=$(printf "%02d" "$INDEX")

    TRAIN="${DATA_ROOT}/random_${ID}.jsonl"
    EXP_NAME="Llama-3.1-8B-SFT-Random-${ID}"
    OUT="${OUTPUT_ROOT}/${EXP_NAME}"
    LOG_FILE="${LOG_ROOT}/${EXP_NAME}.log"
    DONE_FILE="${OUT}/TRAINING_DONE"

    echo
    echo "============================================================"
    echo "开始训练：random_${ID}"
    echo "Train:  $TRAIN"
    echo "Output: $OUT"
    echo "Log:    $LOG_FILE"
    echo "============================================================"

    if [[ ! -f "$TRAIN" ]]; then
        echo "[ERROR] 训练数据不存在：$TRAIN"
        exit 1
    fi

    # 重复执行脚本时，自动跳过已经成功完成的任务。
    # if [[ -f "$DONE_FILE" ]]; then
    #     echo "[SKIP] 开始训练：random_${ID} 已完成"
    #     continue
    # fi

    mkdir -p "$OUT"

    TRAIN_SIZE=$(wc -l < "$TRAIN")
    DEV_SIZE=$(wc -l < "$DEV")

    echo "Training samples:   $TRAIN_SIZE"
    echo "Validation samples: $DEV_SIZE"

    START_TIME=$(date +%s)

    if CUDA_VISIBLE_DEVICES="$GPU_IDS" \
       MASTER_PORT=23457 \
       NPROC_PER_NODE="$NPROC" \
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
        --dataset_num_proc 64 \
        --load_from_cache_file false \
        \
        --loss_scale default \
        \
        --max_length 4096 \
        --truncation_strategy left \
        --packing false \
        \
        --tuner_type lora \
        --target_modules all-linear \
        --lora_rank 16 \
        --lora_alpha 32 \
        --lora_dropout 0.01 \
        --lora_bias none \
        \
        --learning_rate 1e-4 \
        --num_train_epochs 1 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 2 \
        --gradient_accumulation_steps 2 \
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
        --eval_steps 100 \
        \
        --save_strategy steps \
        --save_steps 100 \
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
        \
        --seed 42 \
        --data_seed 42 \
        \
        2>&1 | tee "$LOG_FILE"
    then
        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))

        {
            echo "experiment=$EXP_NAME"
            echo "train=$TRAIN"
            echo "train_size=$TRAIN_SIZE"
            echo "dev=$DEV"
            echo "dev_size=$DEV_SIZE"
            echo "elapsed_seconds=$ELAPSED"
            echo "completed_at=$(date '+%Y-%m-%d %H:%M:%S')"
        } > "$DONE_FILE"

        echo "[DONE] random_${ID} 训练完成，耗时 ${ELAPSED}s"
    else
        EXIT_CODE=$?
        echo "[ERROR] random_${ID} 训练失败，退出码：$EXIT_CODE"
        echo "[ERROR] 查看日志：$LOG_FILE"
        exit "$EXIT_CODE"
    fi
done

echo
echo "============================================================"
echo "全部训练任务完成"
echo "Adapter 范围：random_01 ～ random_12"
echo "输出目录：$OUTPUT_ROOT"
echo "============================================================"
