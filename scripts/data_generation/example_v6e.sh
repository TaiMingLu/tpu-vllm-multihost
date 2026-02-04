#!/bin/bash
set -euo pipefail

# This is an example script showing how to use the sequence KD data generator
# Adapt the paths and parameters to your setup

# Required environment variables
export HF_ACCESS_TOKEN=${HF_ACCESS_TOKEN:?HF_ACCESS_TOKEN is required}

# Dataset configuration
export DATASET_PATH="/path/to/your/parquet/dataset"  # Directory containing *.parquet files
export TEXT_COLUMN="text"  # Column name in parquet files containing text

# Model configuration
# IMPORTANT: This should be a HuggingFace-format model directory
# If you have MaxText checkpoints, you need to convert them first
export MODEL_PATH="/path/to/hf/model"  # e.g., converted from MaxText llama3.1-1b
export TOKENIZER_PATH="/path/to/tokenizer"  # e.g., Llama-3.1-8B tokenizer

# Generation configuration
export MAX_PREFILL_LENGTH=1024
export MAX_TARGET_LENGTH=4096
export GEN_BATCH_SIZE=128
export TEMPERATURE=0.8
export TOP_P=1.0

# TPU configuration
export TENSOR_PARALLEL_SIZE=4  # Adjust based on your TPU topology
# For v6e-8: can use TP=1,2,4,8
# For v6e-16: can use TP=1,2,4,8,16
# etc.

# Output configuration
export OUTPUT_DIR="/tmp/sequence-kd/output"
export GCS_BUCKET_PATH="/path/to/gcs/bucket/output"  # Final output location

# Run the data generation
bash scripts/data_generation/run_sequence_kd.sh
