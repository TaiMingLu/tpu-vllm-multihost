# Parquet Generation Pipeline with vLLM

This directory contains tools for generating text completions from parquet files using vLLM on TPU.

## Files

- **`generate_from_parquet.py`** - Main Python script for generation
- **`run_generation_simple.sh`** - Shell script to run generation on TPU
- **`create_test_parquet.py`** - Create test data for testing the pipeline
- **`test_generation.sh`** - Simple test script (from earlier testing)

## Quick Start

### 1. Create Test Data (Optional)

If you want to test with sample data first:

```bash
# On your local machine
python3 create_test_parquet.py --output test_prompts.parquet --num-samples 20

# Copy to TPU (via GCS bucket)
gsutil cp test_prompts.parquet gs://your-bucket/data/
```

### 2. Configure the Generation Script

Edit `run_generation_simple.sh` and update these variables:

```bash
# Input/Output paths
INPUT_PARQUET="/home/terry/gcs-bucket/data/prompts.parquet"
OUTPUT_PARQUET="/home/terry/gcs-bucket/data/completions.parquet"

# Column names in your parquet file
PROMPT_COLUMN="prompt"          # Column containing input prompts
COMPLETION_COLUMN="completion"  # Column name for generated text

# Generation parameters (adjust as needed)
TEMPERATURE=0.8
TOP_P=0.95
MAX_TOKENS=512
```

### 3. Run Generation

```bash
# Make script executable
chmod +x run_generation_simple.sh

# Run generation
./run_generation_simple.sh
```

The script will:
1. Copy the Python script to your TPU
2. SSH into the TPU and run generation
3. Save results to the output parquet file

## Usage Examples

### Example 1: Basic Usage

```bash
# Edit the script to set your paths
vim run_generation_simple.sh

# Run it
./run_generation_simple.sh
```

### Example 2: Test on Limited Rows

In `run_generation_simple.sh`, uncomment and set:

```bash
MAX_ROWS="--max-rows 100"  # Only process first 100 rows
```

### Example 3: Direct Python Usage

You can also run the Python script directly on the TPU:

```bash
# SSH into TPU
gcloud compute tpus tpu-vm ssh terry@taiming-v6e-8_000103 \
  --project=vision-mix --zone=europe-west4-a --worker=0

# Activate environment
source ~/work-dir/vllm_env/bin/activate

# Run generation
python3 ~/work-dir/generate_from_parquet.py \
  --input /path/to/input.parquet \
  --output /path/to/output.parquet \
  --model /home/terry/gcs-bucket/ckpts/pretrain_param_only_hf/llama3.1-1b-finewebedu-vanilla-s42 \
  --tokenizer /home/terry/gcs-bucket/HF_HOME/Llama-3.1-8B \
  --temperature 0.8 \
  --top-p 0.95 \
  --max-tokens 512 \
  --max-model-len 2048
```

## Input Parquet Format

Your input parquet file should have at least one column containing prompts. For example:

```python
import pandas as pd

df = pd.DataFrame({
    'prompt': [
        'Write a story about...',
        'Explain how...',
        'What is...'
    ],
    'id': [1, 2, 3],
    'metadata': ['...', '...', '...']
})

df.to_parquet('input.parquet')
```

The script will preserve all input columns and add a new column with completions.

## Output Format

The output parquet will contain:
- All original columns from input
- New `completion` column (or whatever you specify with `--completion-column`)

Example:

| prompt | id | metadata | completion |
|--------|----|-----------|--------------------|
| Write a story... | 1 | ... | Once upon a time... |
| Explain how... | 2 | ... | The process works... |

## Generation Parameters

### Core Parameters

- `--temperature` (default: 0.8) - Controls randomness (0.0 = deterministic, 1.0+ = creative)
- `--top-p` (default: 0.95) - Nucleus sampling threshold
- `--top-k` (default: -1) - Top-k sampling (-1 to disable)
- `--max-tokens` (default: 512) - Maximum tokens to generate per prompt
- `--repetition-penalty` (default: 1.0) - Penalize repetition (>1.0 = less repetition)

### Model Parameters

- `--model` - Path to model directory (HuggingFace format)
- `--tokenizer` - Path to tokenizer (defaults to model path)
- `--max-model-len` (default: 2048) - Maximum sequence length
- `--tensor-parallel-size` (default: 1) - Number of tensor parallel replicas

### Processing Options

- `--max-rows` - Process only first N rows (useful for testing)
- `--compression` - Parquet compression (snappy, gzip, brotli, none)
- `--prompt-column` - Name of input column with prompts
- `--completion-column` - Name for output completion column

## Performance Tips

1. **Batch Size**: vLLM automatically batches requests for optimal throughput. No need to manually batch.

2. **Sequence Length**: Set `--max-model-len` based on your data:
   - Shorter = faster but may truncate long prompts
   - Longer = handles any prompt but slower

3. **Temperature**: Lower temperature (0.0-0.5) for factual tasks, higher (0.7-1.0) for creative tasks

4. **Max Tokens**: Balance output quality with generation speed:
   - Shorter completions = faster
   - Longer completions = more detailed but slower

## Monitoring Progress

The script shows:
- Model loading progress
- Generation progress with speed estimates
- Sample completions
- Summary statistics (length distributions, etc.)

Example output:
```
Generating completions for 1000 prompts...
Processed prompts: 100%|████████| 1000/1000 [01:23<00:00, 12.0it/s]

Summary:
  Total prompts processed: 1000

  Completion length stats (characters):
    Mean: 847.3
    Median: 792.0
    Min: 234
    Max: 2048
```

## Troubleshooting

### Model Loading Fails

```
Error: Model not found at /path/to/model
```

**Solution**: Check that the model path is correct and accessible from the TPU:
```bash
# SSH to TPU and check
gcloud compute tpus tpu-vm ssh terry@taiming-v6e-8_000103 ...
ls -la /home/terry/gcs-bucket/ckpts/pretrain_param_only_hf/llama3.1-1b-finewebedu-vanilla-s42
```

### Parquet File Not Found

```
Error: Input path does not exist
```

**Solution**: Ensure your GCS bucket is mounted and the file exists:
```bash
# Check if bucket is mounted
ls /home/terry/gcs-bucket/

# Check if file exists
ls -la /home/terry/gcs-bucket/data/input.parquet
```

### OOM (Out of Memory) Error

```
Error: Out of memory
```

**Solution**: Reduce memory usage:
- Decrease `--max-model-len` (e.g., from 2048 to 1024)
- Decrease `--max-tokens` (e.g., from 512 to 256)
- Process in smaller batches using `--max-rows`

### Prompt Column Not Found

```
Error: Prompt column 'prompt' not found
```

**Solution**: Specify the correct column name:
```bash
# Check available columns
python3 -c "import pandas as pd; df = pd.read_parquet('input.parquet'); print(df.columns)"

# Use correct column name
python3 generate_from_parquet.py --prompt-column "text" ...
```

## Example Workflow

Complete example workflow:

```bash
# 1. Create test data
python3 create_test_parquet.py \
  --output test_data.parquet \
  --num-samples 50

# 2. Copy to GCS bucket (via mounted directory on TPU)
# The TPU should already have the bucket mounted at /home/terry/gcs-bucket
# So you can directly write to it, or use gsutil:
gsutil cp test_data.parquet gs://your-bucket/data/

# 3. Edit generation script
vim run_generation_simple.sh
# Update INPUT_PARQUET and OUTPUT_PARQUET paths

# 4. Run generation
./run_generation_simple.sh

# 5. Download results (if needed)
gsutil cp gs://your-bucket/data/completions.parquet ./

# 6. Inspect results
python3 -c "
import pandas as pd
df = pd.read_parquet('completions.parquet')
print(df.head())
print(f'\nGenerated {len(df)} completions')
"
```

## Advanced: Processing Large Datasets

For very large datasets (millions of rows):

```bash
# Option 1: Process in chunks
for i in {0..9}; do
    START=$((i * 100000))
    python3 generate_from_parquet.py \
      --input data.parquet \
      --output "output_chunk_${i}.parquet" \
      --max-rows 100000 \
      ...
done

# Option 2: Use directory of parquets
# The script can read from a directory of parquet files
python3 generate_from_parquet.py \
  --input /path/to/parquet_directory/ \
  --output combined_output.parquet \
  ...
```

## Model Configuration

The current setup uses:
- **Model**: `llama3.1-1b-finewebedu-vanilla-s42`
- **Architecture**: Custom expansion model with `head_dim=128`
- **Tokenizer**: Llama 3.1 8B tokenizer (compatible)

To use a different model:
1. Convert the MaxText checkpoint to HuggingFace format
2. Update MODEL_PATH in `run_generation_simple.sh`
3. Update TOKENIZER_PATH if needed

## References

- vLLM Documentation: https://docs.vllm.ai/
- Model architecture notes: See `MODEL_ARCHITECTURE_NOTES.md`
- TPU setup: See main project README
