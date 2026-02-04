#!/bin/bash
set -euo pipefail

# Setup all TPU workers with vLLM environment
# Usage: ./setup_multihost.sh <tpu-name>

if [ $# -lt 1 ]; then
    echo "Usage: $0 <tpu-name>"
    echo "Example: $0 my-v5e-16"
    exit 1
fi

TPU_NAME=$1

echo "Setting up all workers on TPU: ${TPU_NAME}"
echo
echo "NOTE: Consider using run_multihost.sh instead - it does setup + run in one call"
echo "      which uses half the API quota."
echo

# Use multihost_runner to run setup on all workers
python3 multihost_runner.py \
    --tpu-name="${TPU_NAME}" \
    --command="bash setup_workers.sh" \
    --script-dir=.

echo
echo "=== All workers setup complete! ==="
echo "You can now run: ./run_multihost.sh ${TPU_NAME}"
