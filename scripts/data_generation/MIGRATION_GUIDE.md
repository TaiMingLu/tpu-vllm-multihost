# Migration Guide: MaxText to vLLM Data Generation

This guide explains the key differences between the MaxText-based and vLLM-based data generation pipelines.

## Architecture Comparison

### MaxText Approach (Original)

```
┌─────────────────────────────────┐
│  maxengine_server (background)  │
│  - Loads model                  │
│  - Serves via JetStream gRPC    │
│  - Port 9000                    │
└─────────────────────────────────┘
          ↑ gRPC
          │
┌─────────────────────────────────┐
│  sequence_kd_parquet.py         │
│  - Reads parquet files          │
│  - Sends async gRPC requests    │
│  - Collects results             │
│  - Saves JSONL                  │
└─────────────────────────────────┘
```

### vLLM Approach (New)

```
┌─────────────────────────────────┐
│  sequence_kd_parquet.py         │
│  - Initializes vLLM engine      │
│  - Reads parquet files          │
│  - Batch generation (in-process)│
│  - Saves JSONL                  │
└─────────────────────────────────┘
```

## Key Differences

| Aspect | MaxText | vLLM |
|--------|---------|------|
| **Server** | Separate maxengine_server | In-process vLLM engine |
| **Communication** | gRPC (async) | Direct Python API |
| **Model Format** | Orbax/Zarr3 checkpoints | HuggingFace format |
| **Batching** | Manual async batching | Automatic by vLLM |
| **Startup** | Server warmup needed | Direct initialization |
| **Dependencies** | JetStream, gRPC | vLLM only |

## Migration Steps

### 1. Convert Model to HuggingFace Format

Your MaxText checkpoints need to be converted:

```bash
# First: Convert to param-only checkpoint (if not done)
python3 -m MaxText.generate_param_only_checkpoint \
  MaxText/configs/base.yml \
  load_full_state_path=gs://bucket/checkpoints/full/items \
  checkpoint_dir=gs://bucket/checkpoints/param_only \
  model_name=llama3.1-1b

# Second: Convert to HuggingFace format
python3 -m MaxText.llama_mistral_mixtral_orbax_to_hf \
  --orbax-checkpoint gs://bucket/checkpoints/param_only/0/items \
  --hf-model-path /path/to/output/hf_model \
  --model-size llama3.1-1b
```

### 2. For Custom Model Sizes

If you have custom architectures (05b, 1b, 3b), ensure the HuggingFace `config.json` matches:

**llama3.1-05b** (base_emb_dim=1024, 12 layers):
```json
{
  "architectures": ["LlamaForCausalLM"],
  "hidden_size": 1024,
  "intermediate_size": 8192,
  "num_attention_heads": 32,
  "num_hidden_layers": 12,
  "num_key_value_heads": 8,
  "vocab_size": 128256,
  "rms_norm_eps": 1e-5,
  "rope_theta": 500000,
  "model_type": "llama"
}
```

**llama3.1-1b** (base_emb_dim=2048, 16 layers):
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

**llama3.1-3b** (base_emb_dim=3072, 28 layers):
```json
{
  "architectures": ["LlamaForCausalLM"],
  "hidden_size": 3072,
  "intermediate_size": 8192,
  "num_attention_heads": 32,
  "num_hidden_layers": 28,
  "num_key_value_heads": 8,
  "vocab_size": 128256,
  "rms_norm_eps": 1e-5,
  "rope_theta": 500000,
  "model_type": "llama"
}
```

### 3. Update Your Scripts

**Old (MaxText):**
```bash
# Start server
python3 -m MaxText.maxengine_server MaxText/configs/base.yml \
  model_name=llama3.1-1b \
  load_parameters_path=gs://bucket/ckpt/items \
  ...

# Run generation
python3 -m MaxText.sequence_kd_parquet \
  --jetstream-server-port 9000 \
  ...
```

**New (vLLM):**
```bash
# Just run generation (no server needed)
export MODEL_PATH="/path/to/hf/model"
bash scripts/data_generation/run_sequence_kd.sh
```

### 4. Configuration Mapping

| MaxText Config | vLLM Config | Notes |
|----------------|-------------|-------|
| `TEACHER_PARAMETERS_PATH` | `MODEL_PATH` | Must be HuggingFace format |
| `SERVER_PER_DEVICE_BATCH` | `GEN_BATCH_SIZE` | vLLM handles batching |
| `ici_tensor_parallelism` | `TENSOR_PARALLEL_SIZE` | Same concept |
| `decode_sampling_temperature` | `TEMPERATURE` | Same |
| `JETSTREAM_SERVER_PORT` | N/A | No server needed |

### 5. Verify Model Format

Before running, verify your model:

```bash
python scripts/data_generation/verify_model.py \
  --model-path /path/to/hf/model
```

Should output:
```
✓ config.json
✓ tokenizer_config.json
✓ Found .safetensors weights
✓ Model appears to be in valid HuggingFace format
✓ Ready for vLLM
```

## Example: Complete Migration

**Original MaxText Setup:**
```bash
# Train in MaxText
python3 -m MaxText.train MaxText/configs/base.yml model_name=llama3.1-1b ...

# Convert to param-only
python3 -m MaxText.generate_param_only_checkpoint ...

# Run data generation with maxengine_server
bash train/data/full_loop_single_v6eu.sh
```

**New vLLM Setup:**
```bash
# 1. Convert checkpoint to HuggingFace
python3 -m MaxText.llama_mistral_mixtral_orbax_to_hf \
  --orbax-checkpoint gs://bucket/ckpt/items \
  --hf-model-path /home/user/models/llama3.1-1b-hf \
  --model-size llama3.1-1b

# 2. Verify model
python scripts/data_generation/verify_model.py \
  --model-path /home/user/models/llama3.1-1b-hf

# 3. Set environment variables
export MODEL_PATH="/home/user/models/llama3.1-1b-hf"
export TOKENIZER_PATH="/home/user/tokenizers/Llama-3.1-8B"
export DATASET_PATH="/home/user/data/finewebedu"
export GCS_BUCKET_PATH="/home/user/output"
export TENSOR_PARALLEL_SIZE=4

# 4. Run data generation
bash scripts/data_generation/run_sequence_kd.sh
```

## Preserved Features

The following features work exactly the same way:

✓ Resume capability (same chunking logic)
✓ Row range tracking
✓ Distributed processing (shuffle files/chunks)
✓ Output format (identical JSONL structure)
✓ GCS bucket integration
✓ Incremental saving

## Troubleshooting

### "Model not found"
- Ensure model is in HuggingFace format
- Run `verify_model.py` to check

### "Out of memory"
- Reduce `GEN_BATCH_SIZE`
- Reduce `MAX_TARGET_LENGTH`
- Increase `TENSOR_PARALLEL_SIZE`

### "No generations"
- Check `TEMPERATURE` and `TOP_P` settings
- Verify tokenizer matches model

### "Slow generation"
- Increase `TENSOR_PARALLEL_SIZE` if you have more cores
- Increase `GEN_BATCH_SIZE` if memory allows
- Check TPU utilization

## Performance Notes

- vLLM typically has **better throughput** than MaxText for inference
- **No server warmup delay** (faster startup)
- **Lower memory overhead** (no separate server process)
- **Easier debugging** (single process, no gRPC)
