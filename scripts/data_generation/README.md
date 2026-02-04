# Sequence KD Data Generation with vLLM

This directory contains scripts for generating sequence-level knowledge distillation data using vLLM on TPU.

## Overview

The data generation pipeline:
1. Reads text data from parquet files
2. Tokenizes and truncates to a prefix length
3. Generates completions using a teacher model (via vLLM)
4. Saves results to JSONL files with format: `{parquet_file, row_idx, prefix, generated}`

## Features

- **Resume capability**: Tracks completed row ranges and only processes missing chunks
- **Distributed processing**: Shuffles files and chunks for parallel processing across multiple instances
- **Incremental saving**: Saves chunks periodically to avoid data loss
- **GCS integration**: Copies results to GCS bucket path

## Requirements

```bash
pip install vllm pandas pyarrow transformers tqdm
```

## Model Preparation

vLLM requires models in HuggingFace format. If you have MaxText checkpoints, you need to convert them first.

### Converting MaxText Checkpoints to HuggingFace Format

If you trained custom models in MaxText (e.g., llama3.1-1b, llama3.1-3b, llama3.1-4b-width), you need to:

1. **Convert to parameter-only checkpoint** (if not already done):
   ```bash
   # See MaxText documentation for generate_param_only_checkpoint.py
   ```

2. **Convert to HuggingFace format**:
   ```bash
   # Use MaxText's conversion script
   python3 -m MaxText.llama_mistral_mixtral_orbax_to_hf \
     --orbax-checkpoint gs://bucket/checkpoints/run/items/12345 \
     --hf-model-path /path/to/output \
     --model-size llama3.1-1b  # or your custom config
   ```

3. **For custom model sizes** (05b, 1b, 3b), you may need to manually create a `config.json`:

   Example for llama3.1-1b (base_emb_dim=2048, num_layers=16):
   ```json
   {
     "architectures": ["LlamaForCausalLM"],
     "hidden_size": 2048,
     "intermediate_size": 8192,
     "num_attention_heads": 32,
     "num_hidden_layers": 16,
     "num_key_value_heads": 8,
     "vocab_size": 128256,
     "rms_norm_eps": 1e-5,
     "rope_theta": 500000,
     "model_type": "llama"
   }
   ```

   See MIGRATION_GUIDE.md for configs for llama3.1-05b and llama3.1-3b.

## Usage

### Basic Usage

```bash
export MODEL_PATH="/path/to/hf/model"
export TOKENIZER_PATH="/path/to/tokenizer"
export DATASET_PATH="/path/to/parquet/files"
export GCS_BUCKET_PATH="/path/to/output"
export HF_ACCESS_TOKEN="your_token"  # if needed

bash scripts/data_generation/run_sequence_kd.sh
```

### Configuration Options

Edit the script or set environment variables:

- `MODEL_PATH`: Path to HuggingFace model directory
- `TOKENIZER_PATH`: Path to tokenizer directory
- `DATASET_PATH`: Directory containing `*.parquet` files
- `GCS_BUCKET_PATH`: Output directory for JSONL files
- `TEXT_COLUMN`: Column name in parquet containing text (default: "text")
- `MAX_PREFILL_LENGTH`: Max tokens for prefix (default: 1024)
- `MAX_TARGET_LENGTH`: Max total tokens (default: 4096)
- `GEN_BATCH_SIZE`: Batch size for generation (default: 128)
- `TEMPERATURE`: Sampling temperature (default: 0.8)
- `TOP_P`: Top-p sampling (default: 1.0)
- `TENSOR_PARALLEL_SIZE`: Number of TPU cores for tensor parallelism (default: 1)

### TPU Configuration

For different TPU topologies:
- **v6e-8**: Use `TENSOR_PARALLEL_SIZE=1,2,4,8`
- **v6e-16**: Use `TENSOR_PARALLEL_SIZE=1,2,4,8,16`

### Example for v6e-8

```bash
# Set environment variables
export HF_ACCESS_TOKEN="hf_..."
export MODEL_PATH="/home/user/models/llama3.1-1b-hf"
export TOKENIZER_PATH="/home/user/tokenizers/Llama-3.1-8B"
export DATASET_PATH="/home/user/data/finewebedu/sample/100BT"
export GCS_BUCKET_PATH="/home/user/gcs-bucket/sequence_kd_data/output"
export TENSOR_PARALLEL_SIZE=4
export GEN_BATCH_SIZE=128

# Run
bash scripts/data_generation/run_sequence_kd.sh
```

## Output Format

Each JSONL file contains one JSON object per line:

```json
{
  "parquet_file": "file_0001.parquet",
  "row_idx": 42,
  "prefix": "Once upon a time",
  "generated": " there was a young prince who lived in a castle..."
}
```

Files are named: `<parquet_basename>_rows_<start>_<end>.jsonl`

Example: `file_0001_rows_0000000_0000512.jsonl`

## Resume Logic

The script automatically detects completed chunks by scanning the output directory for existing JSONL files. If you run the script multiple times:

1. It will skip already-completed row ranges
2. Process only missing chunks
3. Allow multiple instances to run in parallel (they'll coordinate via the filesystem)

## Differences from MaxText Version

- **No separate server**: vLLM runs in-process, no need for JetStream server
- **Simpler setup**: Just run the script, no server warmup needed
- **Model format**: Requires HuggingFace format instead of MaxText/Orbax checkpoints
- **Batching**: vLLM handles batching internally, no async gRPC needed

## Troubleshooting

### Out of Memory

Reduce batch size or max length:
```bash
export GEN_BATCH_SIZE=64
export MAX_TARGET_LENGTH=2048
```

### Model Not Found

Ensure model is in HuggingFace format with proper `config.json`, weights, and tokenizer files.

### TPU Not Detected

Check JAX/vLLM installation:
```bash
python -c "import jax; print(jax.devices())"
```
