#!/bin/bash

CODE_ROOT=$(dirname "$(dirname "$(readlink -f "$0")")")

# MODEL_PATH="${1:-/home/models/DeepSeek-V2-Lite-Chat/}"
# UCM_SPARSE_CONFIG="${2:-${CODE_ROOT}/eval/ucm_sparse_config.json}"
# TEST_DATA_DIR="${3:-${CODE_ROOT}/eval/datasets/LongBench}"

MODEL_PATH="${1}"
UCM_SPARSE_CONFIG="${2}"
TEST_DATA_DIR="${3}"

export PYTHONPATH="$PYTHONPATH:/home/externals/wangwenxin21/vllm:/home/externals/wangwenxin21/unified-cache-management"

export VLLM_VERSION="0.9.2"
export VLLM_USE_V1="1"

# Model 
export MODEL_PATH
MODEL_NAME=$(basename "$MODEL_PATH")
echo "MODEL_PATH is ${MODEL_PATH}"

# Dataset and storage path
STORAGE_BACKENDS="${CODE_ROOT}/ucm_kv_cache/${MODEL_NAME}"
export STORAGE_BACKENDS

SAVE_PATH="${CODE_ROOT}/ucm_sparse_results"
DATASET="LongBench" 
DATASET_SAVE_DIR="${SAVE_PATH}/${MODEL_NAME}/${DATASET}"

mkdir -p "$STORAGE_BACKENDS" "$DATASET_SAVE_DIR" || { echo "Failed to create dirs!"; exit 1; }

DATASET_LOWER="${DATASET,,}"

# -------------------------- LongBench --------------------------
TARGET_FILES=(
"${TEST_DATA_DIR}/multifieldqa_zh.jsonl"
# "${TEST_DATA_DIR}/dureader.jsonl"
)

EXISTING_FILES=()
declare -A seen_files 
for file in "${TARGET_FILES[@]}"; do
    if [[ -f "$file" && -z "${seen_files[$file]}" ]]; then
        seen_files["$file"]=1
        EXISTING_FILES+=("$file")
    fi
done
if [[ ${#EXISTING_FILES[@]} -eq 0 ]]; then
    echo "Error: No valid data files found for '$DATASET'!"
    exit 1
fi

echo -e "\nFound ${#EXISTING_FILES[@]} data files for $DATASET:"
for file in "${EXISTING_FILES[@]}"; do
    rel_path="${file#${BASE_DATA_DIR}/${DATASET}/}"
    echo "  - $rel_path"
done


UCM_CONFIG_NAME=$(basename "$UCM_SPARSE_CONFIG") 
UCM_CONFIG_NAME_NO_EXT="${UCM_CONFIG_NAME%.*}" 

for DATASET_FLIE in "${EXISTING_FILES[@]}"; do
    filename=$(basename "$DATASET_FLIE")
    file_name_no_ext="${filename%.*}"
    export DATASET_FLIE
    export file_name_no_ext
    
    RES_FILE="${DATASET_SAVE_DIR}/${file_name_no_ext}_${UCM_CONFIG_NAME_NO_EXT}.jsonl"
    export RES_FILE
    [[ -f "$RES_FILE" ]] && > "$RES_FILE"

    export UCM_SPARSE_CONFIG
    echo -e "\n======================================"
    echo "Using Config: $UCM_SPARSE_CONFIG"
    echo "======================================"

    python3 "${CODE_ROOT}/eval/inference.py" \

    if [[ ! -f "$RES_FILE" ]]; then
        echo "Warning: test finished but result file not found!"
        continue
    fi
   

    echo -e "\nCalculating F1 score..."
    F1_FILE="${RES_FILE}.f1.txt"
    > /tmp/scores
   
   echo "${CODE_ROOT}/eval/eval.py"
   python3 "${CODE_ROOT}/eval/eval.py" \
        --answer "$RES_FILE" \
        --dataset "$file_name_no_ext" 2>&1 | grep -E "50 score:|All score:" > /tmp/scores

    if [[ -s /tmp/scores ]]; then
        echo "Result file: $RES_FILE" > "$F1_FILE"
        cat /tmp/scores >> "$F1_FILE"
        echo "" >> "$F1_FILE"
        echo "F1 score saved to: $F1_FILE"
        echo -e "\n\n======================================"
        echo ""
        cat "$UCM_SPARSE_CONFIG"
        echo
        cat "$F1_FILE"
        echo "======================================"
    else
        echo "Warning: No valid F1 score generated!"
        touch "$F1_FILE"
    fi

done

rm -rf ${STORAGE_BACKENDS}