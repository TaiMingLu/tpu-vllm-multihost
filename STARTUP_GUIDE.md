# Complete Startup Guide for Brand New TPU

This guide walks you through setting up a completely fresh TPU v6e-32 and running generation.

## Prerequisites

On your **local machine**, ensure you have:
- `gcloud` CLI installed and configured
- Git access to your repo
- GCP project and zone configured:
  ```bash
  gcloud config set project <your-project>
  gcloud config set compute/zone <your-zone>
  ```

## Step-by-Step Setup

### 1. Clone Repository on Local Machine

```bash
cd /scratch/gpfs/ZHUANGL/tl0463/language
git clone https://github.com/TaiMingLu/tpu-vllm.git tpu-inference
cd tpu-inference
```

### 2. Get Your TPU Name

```bash
# List your TPUs
gcloud compute tpus tpu-vm list

# Note your TPU name (e.g., "terry-v6e-32")
export TPU_NAME=your-tpu-name-here
```

### 3. Install vLLM on All TPU Workers (One-Time Setup)

```bash
# This installs Python 3.12, venv, and vllm-tpu on ALL workers
./setup_multihost.sh ${TPU_NAME}
```

**What happens:**
- Copies setup script to all workers
- On each worker:
  - Installs Python 3.12
  - Creates `~/work-dir/vllm_env`
  - Installs `vllm-tpu`
- Takes ~5-10 minutes

**Expected output:**
```
Setting up all workers on TPU: terry-v6e-32

TPU: terry-v6e-32
Project: your-project
Zone: europe-west4-a

Found 4 workers

Copying /path/to/tpu-inference to 4 workers...
Running command on 4 workers...
[120.5s] 4/4 workers completed...
All workers completed successfully!

=== All workers setup complete! ===
You can now run: ./run_multihost.sh terry-v6e-32
```

### 4. Run Generation

```bash
# Provide TPU name and chip count
./run_multihost.sh ${TPU_NAME} 32
```

**What happens:**
- Uses TENSOR_PARALLEL_SIZE=32 (for v6e-32)
- Copies your repo to all workers
- Runs generation script on all workers simultaneously
- vLLM coordinates across hosts automatically

**Expected output:**
```
Running on TPU: terry-v6e-32
Tensor parallel size: 32

Copying /path/to/tpu-inference to 4 workers...
Running command on 4 workers...

=== Sequence KD Config (vLLM Multi-Host) ===
Run name: sequence-kd-vllm-v6e-multi
Tensor parallel size: 32
...
```

## Complete Command Summary

For a **brand new TPU**, run these commands in order:

```bash
# 1. On local machine - clone repo
cd /scratch/gpfs/ZHUANGL/tl0463/language
git clone https://github.com/TaiMingLu/tpu-vllm.git tpu-inference
cd tpu-inference

# 2. Get TPU name
export TPU_NAME=$(gcloud compute tpus tpu-vm list --format="value(name)" | head -1)
echo "Using TPU: ${TPU_NAME}"

# 3. ONE-TIME: Install vLLM on all workers
./setup_multihost.sh ${TPU_NAME}

# 4. Run generation (do this every time)
./run_multihost.sh ${TPU_NAME} 32
```

## Configuration

Before running, you may want to edit `full_loop_vllm_v6e_multi.sh` to adjust:

- **Dataset path**: `DATASET_PATH="/home/terry/gcs-bucket/HF_HOME/finewebedu/sample/100BT"`
- **Model path**: `MODEL_PATH="/home/terry/gcs-bucket/ckpts/..."`
- **Output path**: `GCS_BUCKET_PATH="/home/terry/gcs-bucket/sequence_kd_data/..."`
- **Batch size**: `GEN_BATCH_SIZE=2000`
- **Temperature**: `TEMPERATURE=1.2`
- **Penalties**: `REPETITION_PENALTY=1.5`, etc.

## Verification

### Check if setup worked:

```bash
# SSH to worker-0
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --worker=0

# Check Python and vLLM
cd ~/work-dir
source vllm_env/bin/activate
python3.12 --version  # Should show Python 3.12.x
pip show vllm         # Should show vllm-tpu version
```

### Monitor during generation:

```bash
# Worker-0 output is shown automatically
# For other workers, check logs on your local machine:
cat /tmp/multihost_worker_1.log
cat /tmp/multihost_worker_2.log
cat /tmp/multihost_worker_3.log
```

## Troubleshooting

### Setup fails on one worker:
```bash
# Check logs
cat /tmp/multihost_worker_*.log

# Or SSH to specific worker
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --worker=1
```

### Generation fails:
```bash
# Check if venv exists on all workers
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --worker=all --command="ls ~/work-dir/vllm_env"

# Re-run setup if needed
./setup_multihost.sh ${TPU_NAME}
```


## After First Setup

You only need to run setup once. After that:

```bash
# Every time you want to run generation:
cd /scratch/gpfs/ZHUANGL/tl0463/language/tpu-inference
./run_multihost.sh ${TPU_NAME} 32
```

## Updating Code

If you update the Python script:

```bash
# On local machine
cd /scratch/gpfs/ZHUANGL/tl0463/language/tpu-inference
git pull  # or edit files locally

# Run again - multihost_runner copies updated files automatically
./run_multihost.sh ${TPU_NAME} 32
```

No need to re-run setup unless you need to reinstall vLLM.
