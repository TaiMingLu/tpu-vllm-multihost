# Multi-Host TPU Execution Guide

This directory contains tools for running vLLM inference on multi-host TPU pods.

## Files

- **`multihost_runner.py`**: Simplified multihost runner that copies code and runs commands on all TPU workers
- **`setup_multihost.sh`**: One-time setup script to install vLLM on all workers
- **`setup_workers.sh`**: Worker setup script (called by setup_multihost.sh)
- **`full_loop_vllm_v6e_multi.sh`**: Multi-host version of the generation script
- **`run_multihost.sh`**: Simple launcher script (easiest way to run)

## Quick Start

### Step 0: One-Time Setup (Install vLLM on All Workers)

**Do this once per TPU:**

```bash
cd /path/to/tpu-inference

# Install vLLM and dependencies on all workers
./setup_multihost.sh <tpu-name>
```

This will:
- Create Python venv on each worker
- Install vLLM with TPU support
- Install dependencies (transformers, pandas, etc.)
- Run in parallel on all workers

**Note:** You only need to do this once. The environment persists across runs.

### Step 1: Run Generation

Once setup is complete, run generation:

### Option 1: Simple Launcher (Recommended)

```bash
# From your local machine
cd /path/to/tpu-inference

# Provide TPU name and chip count
./run_multihost.sh <tpu-name> <num-chips>
```

**Examples:**
```bash
./run_multihost.sh terry-v6e-32 32
./run_multihost.sh my-v5e-16 16
```

### Option 2: Direct multihost_runner Usage

```bash
export TPU_NAME=your-tpu-name
export TENSOR_PARALLEL_SIZE=16  # Total chips across all hosts

python3 multihost_runner.py \
  --tpu-name=${TPU_NAME} \
  --command="export TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE} && bash full_loop_vllm_v6e_multi.sh" \
  --script-dir=.
```

## How It Works

1. **`multihost_runner.py`** copies the entire directory to all TPU workers
2. Runs the command on all workers **simultaneously**
3. **vLLM** automatically coordinates across hosts via JAX
4. Only worker-0 outputs to stdout, other workers log to `/tmp/multihost_worker_N.log`

## Configuration

Edit `full_loop_vllm_v6e_multi.sh` to configure:

- Dataset path
- Model path
- Tokenizer path
- Generation parameters (temperature, penalties, etc.)
- Batch size
- Output directory

## Tensor Parallel Size

Set this to the **total number of chips** across **all hosts**:

| TPU Type | Tensor Parallel Size |
|----------|---------------------|
| v5e-16   | 16                  |
| v5e-32   | 32                  |
| v5e-64   | 64                  |
| v6e-16   | 16                  |
| v6e-32   | 32                  |

## Requirements

- gcloud CLI configured with project and zone:
  ```bash
  gcloud config set project <project>
  gcloud config set compute/zone <zone>
  ```

- SSH access to TPU workers:
  ```bash
  gcloud compute tpus tpu-vm ssh <tpu-name> --worker=0
  ```

## Troubleshooting

**Check logs for non-zero workers:**
```bash
cat /tmp/multihost_worker_1.log
cat /tmp/multihost_worker_2.log
```

**Verify TPU workers:**
```bash
gcloud compute tpus tpu-vm describe <tpu-name> --format="value(networkEndpoints.length())"
```

**SSH to specific worker:**
```bash
gcloud compute tpus tpu-vm ssh <tpu-name> --worker=0
```

## Differences from Single-Host

| Aspect | Single-Host | Multi-Host |
|--------|-------------|------------|
| Script | `full_loop_vllm_v6e.sh` | `full_loop_vllm_v6e_multi.sh` |
| Tensor Parallel | 1 | 16, 32, 64, etc. |
| Launcher | Run directly on TPU | Use `multihost_runner.py` |
| Execution | One command | Runs on all workers simultaneously |
