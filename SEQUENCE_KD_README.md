# Sequence KD Data Generation with vLLM

This directory contains the vLLM port of your MaxText sequence KD data generation pipeline.

## Files

- **`sequence_kd_parquet_vllm.py`** - Main generation script (adapted from MaxText version)
- **`full_loop_vllm_v6e.sh`** - Shell script to run generation on TPU (adapted from your original)

## Comparison with MaxText Version

### MaxText Version
- Uses JetStream server (separate process)
- Async gRPC communication
- Server needs warmup time
- Requires specific parallelism flags

### vLLM Version
- Direct model loading (no server)
- Synchronous batched generation
- Immediate inference after model load
- Simplified parallelism config

### Identical Features
- ✅ Same input/output format (parquet → JSONL)
- ✅ Same chunking and resumption logic
- ✅ Same row range tracking
- ✅ Same shuffling for distributed processing
- ✅ Same tokenization and truncation
- ✅ Same output structure: `{parquet_file, row_idx, prefix, generated}`

## Configuration (Matching Your Original)

```bash
# Dataset
DATASET_PATH="/home/terry/gcs-bucket/HF_HOME/finewebedu/sample/100BT"
TEXT_COLUMN="text"

# Model
MODEL_PATH="/home/terry/gcs-bucket/ckpts/pretrain_param_only_hf/llama3.1-1b-finewebedu-vanilla-s42"
TOKENIZER_PATH="/home/terry/gcs-bucket/HF_HOME/Llama-3.1-8B"

# Generation parameters
MAX_PREFILL_LENGTH=1024      # Max prefix tokens (same as MaxText)
MAX_TARGET_LENGTH=4096       # Max total tokens (same as MaxText)
GEN_BATCH_SIZE=128           # Batch size (same as MaxText)
TEMPERATURE=0.8              # Sampling temperature (from decode_sampling_temperature)
TOP_P=0.95                   # Top-p sampling

# Output
OUTPUT_DIR="/tmp/sequence-kd/output"
GCS_BUCKET_PATH="/home/terry/gcs-bucket/sequence_kd_data/finewebedu/sample-100BT/vllm-T50BS42"
```

## Usage

### Quick Start

**On your TPU**, run:

```bash
# Clone the repo
rm -rf ~/tpu-inference
git clone https://github.com/TaiMingLu/tpu-vllm.git ~/tpu-inference
cd ~/tpu-inference

# Run the script
./full_loop_vllm_v6e.sh
```

The script will:
1. Activate vLLM environment
2. Create necessary directories
3. Run generation with exact same parameters as MaxText version
4. Save results to GCS bucket

### Direct Python Usage

If you prefer to run directly on the TPU:

```bash
# SSH to TPU
gcloud compute tpus tpu-vm ssh terry@taiming-v6e-8_000103 \
  --project=vision-mix --zone=europe-west4-a --worker=0

# Activate environment
source ~/work-dir/vllm_env/bin/activate
cd ~/work-dir

# Run generation
python3 -u sequence_kd_parquet_vllm.py \
  --input-dir /home/terry/gcs-bucket/HF_HOME/finewebedu/sample/100BT \
  --output-dir /tmp/sequence-kd/output \
  --model-path /home/terry/gcs-bucket/ckpts/pretrain_param_only_hf/llama3.1-1b-finewebedu-vanilla-s42 \
  --tokenizer-path /home/terry/gcs-bucket/HF_HOME/Llama-3.1-8B \
  --text-column text \
  --batch-size 128 \
  --max-prefill-length 1024 \
  --max-target-length 4096 \
  --temperature 0.8 \
  --top-p 0.95 \
  --gcs-bucket-path /home/terry/gcs-bucket/sequence_kd_data/finewebedu/sample-100BT/vllm-T50BS42
```

## How It Works

### 1. File Discovery
```
Found 1247 parquet files
Processing 1247 files (shuffled)
```
Finds all `.parquet` files in the input directory and shuffles them for distributed processing.

### 2. Resumption Logic
```
Processing: train-00042.parquet
Total rows: 10240, Chunk size: 512
Completed: 5120/10240 rows, Missing chunks: 10/20
```
Checks `GCS_BUCKET_PATH` for completed chunks and only processes missing ones.

### 3. Row Range Processing
```
Processing rows 5120-5632
Built 507 requests
Batch 1/4: 128 requests
  Generating 128 completions...
```
Processes missing rows in chunks, batching requests for efficiency.

### 4. Output Format
Each JSONL file contains:
```json
{"parquet_file": "train-00042.parquet", "row_idx": 5120, "prefix": "The quick brown...", "generated": "fox jumped over..."}
{"parquet_file": "train-00042.parquet", "row_idx": 5121, "prefix": "Machine learning...", "generated": "is a subset of..."}
```

### 5. Chunk Naming
```
train-00042_rows_0005120_0005632.jsonl
```
Format: `{original_filename}_rows_{start:07d}_{end:07d}.jsonl`

## Distributed Processing

To run multiple instances in parallel (same as MaxText), **on your TPU**:

```bash
# Terminal 1 (SSH session 1)
cd ~/tpu-inference && ./full_loop_vllm_v6e.sh

# Terminal 2 (SSH session 2)
cd ~/tpu-inference && ./full_loop_vllm_v6e.sh

# Terminal 3 (SSH session 3)
cd ~/tpu-inference && ./full_loop_vllm_v6e.sh
```

Or run across multiple TPU VMs for even faster processing.

Each instance will:
- Shuffle parquet files randomly
- Check for completed chunks
- Process different missing chunks
- Coordinate via filesystem (no conflicts)

## Parameters

### Required
- `--input-dir`: Directory with parquet files
- `--output-dir`: Local temporary output directory
- `--model-path`: Path to vLLM model (HF format)
- `--tokenizer-path`: Path to tokenizer

### Generation
- `--text-column` (default: "text"): Column name with text data
- `--max-prefill-length` (default: 1024): Max prefix tokens
- `--max-target-length` (default: 4096): Max total tokens (prefix + generation)
- `--batch-size` (default: 128): Inference batch size
- `--temperature` (default: 0.8): Sampling temperature
- `--top-p` (default: 0.95): Top-p (nucleus) sampling

### Storage
- `--gcs-bucket-path`: Final output location (GCS bucket mount)
- `--save-every-n-batches` (default: 4): Chunk size = batch_size × this

### Optional
- `--hf-access-token`: HuggingFace token (if needed)
- `--tensor-parallel-size` (default: 1): Tensor parallel replicas

## Output Structure

```
/home/terry/gcs-bucket/sequence_kd_data/finewebedu/sample-100BT/vllm-T50BS42/
├── train-00000_rows_0000000_0000512.jsonl
├── train-00000_rows_0000512_0001024.jsonl
├── train-00000_rows_0001024_0001536.jsonl
...
├── train-00042_rows_0005120_0005632.jsonl
...
└── train-01246_rows_0009728_0010240.jsonl
```

Each JSONL file is self-contained and can be processed independently.

## Monitoring Progress

The script outputs detailed progress:

```
============================================================
Processing: train-00042.parquet
============================================================
Total rows: 10240, Chunk size: 512
Completed: 5120/10240 rows, Missing chunks: 10/20
Loading data from /home/terry/gcs-bucket/.../train-00042.parquet

Processing rows 5120-5632
Built 507 requests
Batch 1/4: 128 requests
  Generating 128 completions...
Batch 2/4: 128 requests
  Generating 128 completions...
Batch 3/4: 128 requests
  Generating 128 completions...
Batch 4/4: 123 requests
  Generating 123 completions...
Saving 507 results to /tmp/sequence-kd/output/train-00042_rows_0005120_0005632.jsonl
Copying to bucket: /home/terry/gcs-bucket/sequence_kd_data/.../train-00042_rows_0005120_0005632.jsonl
```

## Differences from MaxText Version

### Simpler Setup
- **No server process**: vLLM loads model directly
- **No port management**: No need to check if server is listening
- **No warmup wait**: Start generating immediately after model load
- **No cleanup trap**: No background processes to manage

### Performance Considerations

**MaxText (JetStream)**:
- Async gRPC: Can handle multiple concurrent requests
- Server warmup: 60s delay after port opens
- Parallelism: Complex ICI flags (data/tensor/fsdp/autoregressive)

**vLLM**:
- Synchronous batching: Processes one batch at a time
- Immediate start: Generate as soon as model loads
- Parallelism: Simple tensor_parallel_size flag

**Expected throughput**: Should be similar for batch_size=128, as both use efficient TPU batching.

## Troubleshooting

### Model Load Fails
```
Error: Model not found
```
**Solution**: Check MODEL_PATH is correct on TPU:
```bash
ls -la /home/terry/gcs-bucket/ckpts/pretrain_param_only_hf/llama3.1-1b-finewebedu-vanilla-s42
```

### Parquet Files Not Found
```
No parquet files found in /path
```
**Solution**: Check DATASET_PATH and that files exist:
```bash
ls -la /home/terry/gcs-bucket/HF_HOME/finewebedu/sample/100BT/*.parquet | head
```

### OOM Errors
```
Out of memory
```
**Solution**: Reduce batch size or max_target_length:
```bash
# In full_loop_vllm_v6e.sh
GEN_BATCH_SIZE=64        # Reduce from 128
MAX_TARGET_LENGTH=2048   # Reduce from 4096
```

### Output Directory Not Writable
```
Permission denied
```
**Solution**: Ensure GCS bucket is mounted:
```bash
# Check mount
df -h | grep gcs-bucket

# Check write permissions
touch /home/terry/gcs-bucket/test && rm /home/terry/gcs-bucket/test
```

## Example Output

Sample JSONL entry:
```json
{
  "parquet_file": "train-00042.parquet",
  "row_idx": 5120,
  "prefix": "The concept of machine learning has revolutionized the field of artificial intelligence by enabling computers to learn from data without being explicitly programmed. This approach has led to significant",
  "generated": " advances in various domains, including computer vision, natural language processing, and robotics. Machine learning algorithms can be broadly categorized into supervised learning, unsupervised learning, and reinforcement learning, each with its own set of applications and challenges. Supervised learning involves training models on labeled data to make predictions, while unsupervised learning focuses on discovering patterns in unlabeled data."
}
```

## Integration with Training

The output JSONL files are ready for training:

```python
import json

# Load generated data
with open('train-00042_rows_0005120_0005632.jsonl') as f:
    for line in f:
        item = json.loads(line)
        prefix = item['prefix']
        generated = item['generated']
        full_text = prefix + generated
        # Use for knowledge distillation training
```

## Performance Expectations

Based on your MaxText setup:
- **Input**: ~20M examples from finewebedu
- **Batch size**: 128
- **Max tokens**: 1024 prefix + up to 3072 generation = 4096 total
- **Chunks**: 512 rows per chunk (128 batch × 4 save intervals)

**Estimated time** (single TPU v6e-8):
- ~1-2 seconds per batch of 128
- ~4-8 seconds per chunk of 512 rows
- ~10-20K rows per hour
- **Total**: ~1000-2000 hours for 20M examples on single host

For faster processing, run multiple instances in parallel across different TPU hosts.
