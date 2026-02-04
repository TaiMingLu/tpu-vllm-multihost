# tpu-vllm-multihost

A quickstart repository for running vLLM inference on Google TPUs, including multi-host support.

> **Note:** This repository may become outdated. For the latest updates and official documentation, please refer to the official repository: https://github.com/vllm-project/tpu-inference
>
> For detailed tpu-inference documentation, see [README_ORIGINAL.md](README_ORIGINAL.md).

## Prerequisites

```bash
export TPU_NAME="your-tpu-name"
export PROJECT_ID="your-project"
export ZONE="your-zone"
export TPU_TYPE="v6e-32"  # or v5e-16, etc.
```

Verify your TPU is ready:

```bash
gcloud compute tpus tpu-vm describe ${TPU_NAME} \
  --project=${PROJECT_ID} --zone=${ZONE}
```

## Single-Host (e.g., v6e-8)

Setup environment (on all workers):

```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --project=${PROJECT_ID} --zone=${ZONE} \
  --worker=all \
  --command='
    sudo apt-get update && sudo apt-get install -y python3.12 python3.12-venv git
    rm -rf ~/work-dir
    mkdir ~/work-dir
    cd ~/work-dir
    python3.12 -m venv vllm_env --symlinks
    source vllm_env/bin/activate
    pip install vllm-tpu ray[default]
    python -c "import vllm; import tpu_inference; print(\"vLLM ready!\")"
  '
```

Run inference (on worker 0):

```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --project=${PROJECT_ID} --zone=${ZONE} \
  --worker=0 \
  --command='
    export HF_TOKEN=<YOUR_TOKEN>
    rm -rf ~/tpu-inference
    git clone https://github.com/TaiMingLu/tpu-vllm-multihost.git ~/tpu-inference
    cd ~/tpu-inference/quickstart
    source ~/work-dir/vllm_env/bin/activate
    python basic_inference.py --tensor-parallel-size 8
  '
```

If running single host on a multihost machine, set by `export TPU_VISIBLE_CHIPS` first.

## Multi-Host (e.g., v6e-32)

The full loop (environment setup, Ray cluster, inference) is handled automatically:

```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --project=${PROJECT_ID} --zone=${ZONE} \
  --worker=0 \
  --command='
    # Generate SSH key once (non-interactive)
    ssh-keygen -t rsa -N "" -f ~/.ssh/google_compute_engine -q <<< y || true
    gcloud config set compute/zone '${ZONE}'

    rm -rf ~/tpu-inference
    git clone https://github.com/TaiMingLu/tpu-vllm-multihost.git ~/tpu-inference
    cd ~/tpu-inference/quickstart
    ./run_multihost.sh '${TPU_NAME}' '${TPU_TYPE}'
  '
```

`run_multihost.sh` handles:
1. Copies code to all workers via `multihost_runner.py`
2. Installs Python 3.12 and vLLM on each worker (if not already installed)
3. Sets `WORKER_ID` and `RAY_HEAD_IP` for each worker
4. Worker 0 starts Ray head, other workers join
5. Worker 0 runs inference with `TPU_MULTIHOST_BACKEND=ray`

## Tensor Parallel Size

| TPU Type | Chips | Single/Multi-Host |
|----------|-------|-------------------|
| v6e-8    | 4     | Single            |
| v6e-16   | 8     | Multi             |
| v6e-32   | 16    | Multi             |
| v6e-8    | 8     | Single            |
| v6e-16   | 16    | Multi             |
| v6e-32   | 32    | Multi             |

## Using a Different Model or Script

```bash
# Single-host
python basic_inference.py --model google/gemma-2b-it --tensor-parallel-size 8

# Multi-host - different model
export MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
./run_multihost.sh ${TPU_NAME} ${TPU_TYPE}

# Multi-host - custom inference script
./run_multihost.sh ${TPU_NAME} ${TPU_TYPE} my_inference.py
```

## Cleanup

```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --project=${PROJECT_ID} --zone=${ZONE} \
  --worker=all \
  --command='
    ray stop --force 2>/dev/null || true
    sudo pkill -9 -f python 2>/dev/null || true
    sudo rm -f /tmp/libtpu_lockfile
  '
```
