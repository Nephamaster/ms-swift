#!/usr/bin/env bash
set -euo pipefail

# Automatically merge:
#   Qwen3/Llama3.1 x Kmeans/Random x ID 01..12
#
# Selection rule:
#   1. Find the largest vN-* directory under each experiment directory.
#   2. Read best_model_checkpoint from the latest checkpoint's trainer_state.json.
#   3. Merge the recorded best checkpoint with the corresponding Base model.

ADAPTER_ROOT="${ADAPTER_ROOT:-/share/project/wuhaiming/spaces/scs/output/adapters}"

MERGED_OUTPUT_ROOT="${MERGED_OUTPUT_ROOT:-/share/project/wuhaiming/spaces/scs/output/models}"

QWEN_BASE_MODEL="${QWEN_BASE_MODEL:-/share/project/wuhaiming/data/models/Qwen3-8B-Base}"
LLAMA_BASE_MODEL="${LLAMA_BASE_MODEL:-/share/project/wuhaiming/data/models/LLama-3.1-8B}"
PYTHON_BIN="${PYTHON_BIN:-python}"

START_INDEX="${START_INDEX:-1}"
END_INDEX="${END_INDEX:-12}"

FAILED=()
SKIPPED=0
MERGED=0

require_dir() {
    local path="$1"
    local description="$2"
    if [[ ! -d "$path" ]]; then
        echo "[ERROR] $description does not exist: $path" >&2
        return 1
    fi
}

find_experiment_dir() {
    local model_kind="$1"
    local data_kind="$2"
    local id="$3"
    local candidate
    local candidates=()

    case "${model_kind}:${data_kind}" in
        qwen3:Kmeans)
            candidates=(
                "Qwen3-8B-Base-SFT-Kmeans-${id}"
                "Qwen3-8B-SFT-Kmeans-${id}"
            )
            ;;
        qwen3:Random)
            candidates=(
                "Qwen3-8B-Base-SFT-Random-${id}"
                "Qwen3-8B-SFT-Random-${id}"
            )
            ;;
        llama31:Kmeans)
            candidates=(
                "LLama-3.1-8B-SFT-Kmeans-${id}"
                "Llama-3.1-8B-SFT-Kmeans-${id}"
            )
            ;;
        llama31:Random)
            candidates=(
                "LLama-3.1-8B-SFT-Random-${id}"
                "Llama-3.1-8B-SFT-Random-${id}"
            )
            ;;
    esac

    for candidate in "${candidates[@]}"; do
        if [[ -d "${ADAPTER_ROOT}/${candidate}" ]]; then
            printf '%s\n' "${ADAPTER_ROOT}/${candidate}"
            return 0
        fi
    done

    echo "[ERROR] No experiment directory found for ${model_kind}/${data_kind}/${id}" >&2
    echo "        Tried: ${candidates[*]}" >&2
    return 1
}

find_latest_run() {
    local experiment_dir="$1"
    local runs

    runs=$(find "$experiment_dir" \
        -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
        | awk '/^v[0-9]+-/' \
        | sort -V)

    if [[ -z "$runs" ]]; then
        echo "[ERROR] No vN-* run directory found: $experiment_dir" >&2
        return 1
    fi

    printf '%s\n' "$runs" | tail -n 1
}

find_latest_checkpoint() {
    local run_dir="$1"
    local checkpoints

    checkpoints=$(find "$run_dir" \
        -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
        | awk '/^checkpoint-[0-9]+$/' \
        | sort -V)

    if [[ -z "$checkpoints" ]]; then
        echo "[ERROR] No checkpoint-* directory found: $run_dir" >&2
        return 1
    fi

    printf '%s\n' "$checkpoints" | tail -n 1
}

find_best_checkpoint() {
    local run_dir="$1"
    local latest_checkpoint
    local trainer_state
    local recorded_path
    local checkpoint_name
    local relocated_path

    latest_checkpoint=$(find_latest_checkpoint "$run_dir") || return 1
    trainer_state="${run_dir}/${latest_checkpoint}/trainer_state.json"

    if [[ ! -f "$trainer_state" ]]; then
        echo "[ERROR] trainer_state.json does not exist: $trainer_state" >&2
        return 1
    fi

    recorded_path=$(
        "$PYTHON_BIN" - "$trainer_state" <<'PY'
import json
import sys

trainer_state_path = sys.argv[1]
with open(trainer_state_path, encoding='utf-8') as file:
    state = json.load(file)

best_checkpoint = state.get('best_model_checkpoint')
if not best_checkpoint:
    raise SystemExit(
        f'[ERROR] best_model_checkpoint is empty in: {trainer_state_path}'
    )

print(best_checkpoint)
PY
    ) || return 1

    checkpoint_name=$(basename "$recorded_path")
    if [[ ! "$checkpoint_name" =~ ^checkpoint-[0-9]+$ ]]; then
        echo "[ERROR] Invalid best_model_checkpoint in $trainer_state: $recorded_path" >&2
        return 1
    fi

    # Prefer the recorded absolute path. If the run directory was moved,
    # resolve the checkpoint by basename inside the selected vN-* directory.
    if [[ -d "$recorded_path" ]]; then
        printf '%s\n' "$recorded_path"
        return 0
    fi

    relocated_path="${run_dir}/${checkpoint_name}"
    if [[ -d "$relocated_path" ]]; then
        printf '%s\n' "$relocated_path"
        return 0
    fi

    echo "[ERROR] Best checkpoint does not exist: $recorded_path" >&2
    echo "        Also tried: $relocated_path" >&2
    return 1
}

has_complete_model() {
    local output_dir="$1"

    [[ -f "${output_dir}/config.json" ]] || return 1
    find "$output_dir" -maxdepth 1 -type f \
        \( -name 'model*.safetensors' -o -name 'pytorch_model*.bin' \) \
        -print -quit | grep -q .
}

get_base_model() {
    local model_kind="$1"

    case "$model_kind" in
        qwen3)
            printf '%s\n' "$QWEN_BASE_MODEL"
            ;;
        llama31)
            # Support the spelling used by the existing loop script as a fallback.
            if [[ -d "$LLAMA_BASE_MODEL" ]]; then
                printf '%s\n' "$LLAMA_BASE_MODEL"
            elif [[ -d "/share/project/wuhaiming/data/models/Llama-3.1-8B" ]]; then
                printf '%s\n' "/share/project/wuhaiming/data/models/Llama-3.1-8B"
            else
                printf '%s\n' "$LLAMA_BASE_MODEL"
            fi
            ;;
        *)
            echo "[ERROR] Unsupported model kind: $model_kind" >&2
            return 1
            ;;
    esac
}

get_output_dir() {
    local model_kind="$1"
    local data_kind="$2"
    local id="$3"

    case "$model_kind" in
        qwen3)
            printf '%s\n' "${MERGED_OUTPUT_ROOT}/Qwen3-8B-Base-SFT-${data_kind}-${id}"
            ;;
        llama31)
            printf '%s\n' "${MERGED_OUTPUT_ROOT}/LLama-3.1-8B-SFT-${data_kind}-${id}"
            ;;
        *)
            return 1
            ;;
    esac
}

merge_one() {
    local model_kind="$1"
    local data_kind="$2"
    local id="$3"
    local experiment_dir
    local latest_run
    local run_dir
    local adapter
    local output_dir
    local base_model
    local model_type
    local template

    echo
    echo "============================================================"
    echo "[START] ${model_kind}/${data_kind}/${id}"
    echo "============================================================"

    experiment_dir=$(find_experiment_dir "$model_kind" "$data_kind" "$id") || return 1
    latest_run=$(find_latest_run "$experiment_dir") || return 1
    run_dir="${experiment_dir}/${latest_run}"
    adapter=$(find_best_checkpoint "$run_dir") || return 1
    output_dir=$(get_output_dir "$model_kind" "$data_kind" "$id") || return 1
    base_model=$(get_base_model "$model_kind") || return 1

    case "$model_kind" in
        qwen3)
            model_type="qwen3"
            template="qwen3"
            ;;
        llama31)
            model_type="llama"
            template="llama3_2"
            ;;
    esac

    echo "Base model : $base_model"
    echo "Experiment : $experiment_dir"
    echo "Run        : $latest_run"
    echo "Adapter    : $adapter"
    echo "Output     : $output_dir"

    require_dir "$base_model" "Base model" || return 1
    require_dir "$adapter" "Adapter checkpoint" || return 1

    if [[ -e "$output_dir" ]]; then
        if has_complete_model "$output_dir"; then
            echo "[SKIP] Complete merged model already exists: $output_dir"
            SKIPPED=$((SKIPPED + 1))
            return 0
        fi

        echo "[ERROR] Output directory exists but is incomplete: $output_dir" >&2
        echo "        It was not deleted. Rename or remove it manually, then rerun." >&2
        return 1
    fi

    mkdir -p "$(dirname "$output_dir")"

    swift export \
        --model "$base_model" \
        --model_type "$model_type" \
        --template "$template" \
        --adapters "$adapter" \
        --output_dir "$output_dir" \
        --merge_lora true \
        --safe_serialization true \
        --max_shard_size 5GB

    if ! has_complete_model "$output_dir"; then
        echo "[ERROR] swift export finished but the output looks incomplete: $output_dir" >&2
        return 1
    fi

    echo "[DONE] ${model_kind}/${data_kind}/${id}"
    MERGED=$((MERGED + 1))
}

require_dir "$ADAPTER_ROOT" "Adapter root" || exit 1
require_dir "$QWEN_BASE_MODEL" "Qwen3 Base model" || exit 1

if [[ ! -d "$LLAMA_BASE_MODEL" && ! -d "/share/project/wuhaiming/data/models/Llama-3.1-8B" ]]; then
    echo "[ERROR] Llama3.1 Base model does not exist: $LLAMA_BASE_MODEL" >&2
    exit 1
fi

for model_kind in qwen3 llama31; do
    for data_kind in Kmeans Random; do
        for index in $(seq "$START_INDEX" "$END_INDEX"); do
            id=$(printf '%02d' "$index")
            if ! merge_one "$model_kind" "$data_kind" "$id"; then
                FAILED+=("${model_kind}/${data_kind}/${id}")
            fi
        done
    done
done

echo
echo "============================================================"
echo "Merge summary"
echo "Merged : $MERGED"
echo "Skipped: $SKIPPED"
echo "Failed : ${#FAILED[@]}"
if [[ ${#FAILED[@]} -gt 0 ]]; then
    printf '  - %s\n' "${FAILED[@]}"
    exit 1
fi
echo "All requested models completed successfully."
