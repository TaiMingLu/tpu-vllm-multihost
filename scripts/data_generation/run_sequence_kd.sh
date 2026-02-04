#!/bin/bash
set -euo pipefail

# Configuration
RUN_NAME=${RUN_NAME:-"sequence-kd-vllm"}
DATASET_PATH=${DATASET_PATH:?DATASET_PATH is required}
DATA_SPLIT=${DATA_SPLIT:-"train"}
TEXT_COLUMN=${TEXT_COLUMN:-"text"}
MODEL_PATH=${MODEL_PATH:?MODEL_PATH is required}
TOKENIZER_PATH=${TOKENIZER_PATH:?TOKENIZER_PATH is required}
MAX_PREFILL_LENGTH=${MAX_PREFILL_LENGTH:-1024}
MAX_TARGET_LENGTH=${MAX_TARGET_LENGTH:-4096}
GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-128}
TEMPERATURE=${TEMPERATURE:-0.8}
TOP_P=${TOP_P:-1.0}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
HF_ACCESS_TOKEN=${HF_ACCESS_TOKEN:-}

OUTPUT_DIR=${OUTPUT_DIR:-"/tmp/sequence-kd/output"}
GCS_BUCKET_PATH=${GCS_BUCKET_PATH:?GCS_BUCKET_PATH is required}

printf '\n=== Sequence KD Config (vLLM) ===\n'
printf 'Run name: %s\n' "$RUN_NAME"
printf 'Dataset: %s (%s)\n' "$DATASET_PATH" "$DATA_SPLIT"
printf 'Model: %s\n' "$MODEL_PATH"
printf 'Tokenizer: %s\n' "$TOKENIZER_PATH"
printf 'Output dir: %s\n' "$OUTPUT_DIR"
printf 'GCS bucket path: %s\n' "$GCS_BUCKET_PATH"
printf 'Batch size: %s\n' "$GEN_BATCH_SIZE"
printf 'Max prefill length: %s\n' "$MAX_PREFILL_LENGTH"
printf 'Max target length: %s\n' "$MAX_TARGET_LENGTH"
printf 'Temperature: %s\n' "$TEMPERATURE"
printf 'Top-p: %s\n' "$TOP_P"
printf 'Tensor parallel size: %s\n' "$TENSOR_PARALLEL_SIZE"
printf '==================================\n\n'

# Create output directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${GCS_BUCKET_PATH}"

# Run data generation
python3 -u scripts/data_generation/sequence_kd_parquet.py \
  --model-path "${MODEL_PATH}" \
  --input-dir "${DATASET_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --tokenizer-path "${TOKENIZER_PATH}" \
  ${HF_ACCESS_TOKEN:+--hf-access-token "${HF_ACCESS_TOKEN}"} \
  --text-column "${TEXT_COLUMN}" \
  --batch-size "${GEN_BATCH_SIZE}" \
  --max-prefill-length "${MAX_PREFILL_LENGTH}" \
  --max-target-length "${MAX_TARGET_LENGTH}" \
  --gcs-bucket-path "${GCS_BUCKET_PATH}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"

echo "Data generation completed!"
