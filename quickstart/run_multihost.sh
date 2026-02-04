#!/bin/bash
set -euo pipefail

# Simple launcher for multi-host TPU execution
# Usage: ./run_multihost.sh <tpu-name> <tpu-type-or-chip-count> [model] [inference-script]
# Example:
#   ./run_multihost.sh my-v6e-32 v6e-32
#   ./run_multihost.sh my-v6e-32 v6e-32 Qwen/Qwen2.5-72B-Instruct
#   ./run_multihost.sh my-v6e-32 32 Qwen/Qwen2.5-72B-Instruct my_inference.py

if [ $# -lt 2 ]; then
    echo "Usage: $0 <tpu-name> <tpu-type-or-chip-count> [model] [inference-script]"
    echo "Examples:"
    echo "  $0 my-v6e-32 v6e-32                                      # Default model"
    echo "  $0 my-v6e-32 v6e-32 Qwen/Qwen2.5-72B-Instruct            # Specify model"
    echo "  $0 my-v6e-32 32 Qwen/Qwen2.5-72B-Instruct my_script.py   # Model + custom script"
    exit 1
fi

TPU_NAME=$1
TPU_TYPE_OR_SIZE=$2
MODEL_PATH="${3:-${MODEL_PATH:-Qwen/Qwen2.5-72B-Instruct}}"
INFERENCE_SCRIPT="${4:-basic_inference.py}"

# Extract chip count if TPU type provided (e.g., "v6e-32" -> "32")
if [[ "$TPU_TYPE_OR_SIZE" =~ ^v[0-9]+[a-z]*-([0-9]+)$ ]]; then
    TENSOR_PARALLEL_SIZE="${BASH_REMATCH[1]}"
    echo "Detected TPU type: ${TPU_TYPE_OR_SIZE} -> ${TENSOR_PARALLEL_SIZE} chips"
else
    TENSOR_PARALLEL_SIZE=$TPU_TYPE_OR_SIZE
fi

echo "Running on TPU: ${TPU_NAME}"
echo "Tensor parallel size: ${TENSOR_PARALLEL_SIZE}"
echo "Model: ${MODEL_PATH}"
echo "Inference script: ${INFERENCE_SCRIPT}"
echo

python3 multihost_runner.py \
    --tpu-name="${TPU_NAME}" \
    --command="export TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE} && export MODEL_PATH='${MODEL_PATH}' && export INFERENCE_SCRIPT='${INFERENCE_SCRIPT}' && bash setup_and_run.sh" \
    --script-dir=.
