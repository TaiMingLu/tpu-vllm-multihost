"""Simple parquet-based sequence KD data generator using vLLM.

Processes parquet files one by one, generates completions, and saves to JSONL.
Each output line contains: {parquet_file, row_idx, prefix, generated}

Example:
  python3 scripts/data_generation/sequence_kd_parquet.py \
    --model-path /path/to/model \
    --input-dir /path/to/parquets \
    --output-dir /path/to/output \
    --tokenizer-path /path/to/tokenizer
"""

import argparse
import json
import os
import glob
import shutil
import random
from dataclasses import dataclass
from typing import List

import pandas as pd
import pyarrow.parquet as pq
import transformers
from tqdm import tqdm

# Import vLLM
from vllm import LLM, SamplingParams


@dataclass
class Request:
    parquet_file: str
    row_idx: int
    prefix_text: str
    prompt_token_ids: List[int]
    max_output_tokens: int


def run_inference(requests, llm, tokenizer, config):
    """Runs batch inference using vLLM."""
    if not requests:
        return []

    # Prepare prompts (as token IDs)
    prompts = [{"prompt_token_ids": req.prompt_token_ids} for req in requests]

    # Prepare sampling params for each request
    sampling_params_list = [
        SamplingParams(
            max_tokens=req.max_output_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            skip_special_tokens=True,
        )
        for req in requests
    ]

    print(f"Generating {len(requests)} completions...")

    # Generate completions
    outputs = llm.generate(
        prompts,
        sampling_params=sampling_params_list[0] if len(set(str(sp) for sp in sampling_params_list)) == 1 else sampling_params_list,
        use_tqdm=True,
    )

    # Process results
    results = []
    for i, output in enumerate(outputs):
        request = requests[i]
        generated = output.outputs[0].text.strip()

        results.append({
            "parquet_file": os.path.basename(request.parquet_file),
            "row_idx": request.row_idx,
            "prefix": request.prefix_text,
            "generated": generated,
        })

    return results


def get_completed_row_ranges(gcs_bucket_path, parquet_basename) -> List[tuple]:
    """Get list of completed (start, end) row ranges for a parquet file."""
    if not gcs_bucket_path or not os.path.exists(gcs_bucket_path):
        return []

    completed = []
    prefix = parquet_basename.replace(".parquet", "_rows_")
    for f in os.listdir(gcs_bucket_path):
        if f.startswith(prefix) and f.endswith(".jsonl"):
            # Extract row range from filename like "file_rows_0000000_0005120.jsonl"
            try:
                range_str = f.replace(prefix, "").replace(".jsonl", "")
                start, end = range_str.split("_")
                completed.append((int(start), int(end)))
            except (ValueError, IndexError):
                pass
    return sorted(completed)


def get_missing_row_ranges(completed_ranges, total_rows, chunk_size) -> List[tuple]:
    """Calculate which row ranges still need processing.

    Uses fixed chunk boundaries (0, chunk_size, 2*chunk_size, ...) but
    only includes actually-missing rows within each chunk.
    """
    # Build set of all completed rows
    completed_rows = set()
    for start, end in completed_ranges:
        for i in range(start, min(end, total_rows)):
            completed_rows.add(i)

    # Use fixed chunk boundaries, but only include missing rows
    missing = []
    for chunk_start in range(0, total_rows, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_rows)

        # Find actually missing rows within this fixed chunk
        missing_in_chunk = [i for i in range(chunk_start, chunk_end) if i not in completed_rows]

        if missing_in_chunk:
            # Use the actual range of missing rows
            actual_start = missing_in_chunk[0]
            actual_end = missing_in_chunk[-1] + 1
            missing.append((actual_start, actual_end))

    return missing


def main(config):
    # Find all parquet files
    parquet_files = sorted(glob.glob(os.path.join(config.input_dir, "*.parquet")))
    if not parquet_files:
        print(f"No parquet files found in {config.input_dir}")
        return

    print(f"Found {len(parquet_files)} parquet files")

    # Load tokenizer
    print(f"Loading tokenizer from {config.tokenizer_path}")
    if os.path.exists(config.tokenizer_path):
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            config.tokenizer_path, local_files_only=True
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            config.tokenizer_path, token=config.hf_access_token
        )

    # Initialize vLLM model
    print(f"Loading model from {config.model_path}")
    llm = LLM(
        model=config.model_path,
        tokenizer=config.tokenizer_path,
        max_model_len=config.max_target_length,
        tensor_parallel_size=config.tensor_parallel_size,
        trust_remote_code=True,
        enforce_eager=config.enforce_eager,
    )

    # Shuffle parquet files for distributed processing
    random.shuffle(parquet_files)
    print(f"Processing {len(parquet_files)} files (shuffled)")

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Chunk size in rows (batch_size * save_every_n_batches)
    chunk_size = config.batch_size * config.save_every_n_batches

    # Process each parquet file in random order
    processed_count = 0
    for parquet_path in parquet_files:
        filename = os.path.basename(parquet_path)

        # Get row count from metadata (instant, no data loading)
        total_rows = pq.ParquetFile(parquet_path).metadata.num_rows

        # Check completed row ranges
        completed_ranges = get_completed_row_ranges(config.gcs_bucket_path, filename)
        missing_ranges = get_missing_row_ranges(completed_ranges, total_rows, chunk_size)

        if not missing_ranges:
            print(f"Skipping {filename} - all {total_rows} rows completed")
            continue

        total_chunks = (total_rows + chunk_size - 1) // chunk_size
        completed_rows = sum(end - start for start, end in completed_ranges)

        print(f"\n{'='*60}")
        print(f"Processing: {filename}")
        print(f"{'='*60}")
        print(f"Total rows: {total_rows}, Chunk size: {chunk_size}")
        print(f"Completed: {completed_rows}/{total_rows} rows, Missing chunks: {len(missing_ranges)}/{total_chunks}")

        # Load full parquet data now that we know we need to process it
        print(f"Loading data from {parquet_path}")
        df = pd.read_parquet(parquet_path)

        if config.text_column not in df.columns:
            print(f"Column '{config.text_column}' not found, skipping")
            continue

        # Shuffle missing ranges so different instances work on different chunks
        random.shuffle(missing_ranges)

        # Process each missing range
        for start_row, end_row in missing_ranges:
            # Re-check if this range was completed by another instance
            current_completed = get_completed_row_ranges(config.gcs_bucket_path, filename)
            if any(s <= start_row and e >= end_row for s, e in current_completed):
                print(f"Rows {start_row}-{end_row} completed by another instance, skipping")
                continue

            print(f"\nProcessing rows {start_row}-{end_row}")

            # Build requests for this row range
            requests = []
            for idx in range(start_row, end_row):
                text = df.iloc[idx][config.text_column]
                if not isinstance(text, str) or not text.strip():
                    continue

                tokens = tokenizer.encode(text, add_special_tokens=False)
                if not tokens:
                    continue

                if len(tokens) > config.max_prefill_length:
                    tokens = tokens[:config.max_prefill_length]

                max_output = config.max_target_length - len(tokens)
                if max_output <= 0:
                    continue

                prefix_text = tokenizer.decode(tokens, skip_special_tokens=True)
                requests.append(Request(
                    parquet_file=parquet_path,
                    row_idx=idx,
                    prefix_text=prefix_text,
                    prompt_token_ids=tokens,
                    max_output_tokens=max_output,
                ))

            if not requests:
                print(f"No valid requests for rows {start_row}-{end_row}")
                continue

            print(f"Built {len(requests)} requests")

            # Process in batches
            all_results = []
            total_batches = (len(requests) + config.batch_size - 1) // config.batch_size

            for i in range(0, len(requests), config.batch_size):
                batch = requests[i:i + config.batch_size]
                batch_num = i // config.batch_size + 1
                print(f"Batch {batch_num}/{total_batches}: {len(batch)} requests")

                results = run_inference(batch, llm, tokenizer, config)
                all_results.extend(results)

            # Save chunk
            chunk_name = filename.replace(".parquet", f"_rows_{start_row:07d}_{end_row:07d}.jsonl")
            temp_file = os.path.join(config.output_dir, chunk_name)

            print(f"Saving {len(all_results)} results to {temp_file}")
            with open(temp_file, "w") as f:
                for result in all_results:
                    f.write(json.dumps(result) + "\n")

            # Copy to bucket
            if config.gcs_bucket_path:
                bucket_file = os.path.join(config.gcs_bucket_path, chunk_name)
                print(f"Copying to bucket: {bucket_file}")
                os.makedirs(config.gcs_bucket_path, exist_ok=True)
                shutil.copy(temp_file, bucket_file)

            processed_count += 1

        print(f"Completed {filename}")

    print(f"\nAll done! Saved {processed_count} chunks this session")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to model (HuggingFace format)")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing parquet files")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save JSONL outputs")
    parser.add_argument("--tokenizer-path", type=str, required=True, help="Path to tokenizer")
    parser.add_argument("--hf-access-token", type=str, default=None, help="HF token if needed")
    parser.add_argument("--text-column", type=str, default="text", help="Column name for text")
    parser.add_argument("--max-prefill-length", type=int, default=1024, help="Max prefix tokens")
    parser.add_argument("--max-target-length", type=int, default=4096, help="Max total tokens")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for inference")
    parser.add_argument("--gcs-bucket-path", type=str, default=None, help="Path to mounted GCS bucket for final output")
    parser.add_argument("--save-every-n-batches", type=int, default=4, help="Save checkpoint every N batches")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--enforce-eager", action="store_true", help="Disable CUDA graphs (for debugging)")

    config = parser.parse_args()
    main(config)
