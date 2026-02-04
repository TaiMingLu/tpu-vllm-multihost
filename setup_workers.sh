#!/bin/bash
set -euo pipefail

# Setup script to install vLLM and dependencies on all TPU workers
# This will be run by multihost_runner on each worker

echo "=== Setting up TPU worker ==="
echo "Worker: $(hostname)"
echo

# Install Python 3.12
echo "Installing Python 3.12..."

# Wait for any running apt processes to finish (unattended-upgrades)
echo "Waiting for apt lock..."
while sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 ; do
    echo "  Waiting for other apt process to finish..."
    sleep 5
done

sudo apt-get update && sudo apt-get install -y python3.12 python3.12-venv

# Clean and create work directory
echo "Setting up work directory..."
rm -rf ~/work-dir
mkdir ~/work-dir
cd ~/work-dir

# Create Python virtual environment with symlinks
echo "Creating Python 3.12 virtual environment..."
python3.12 -m venv vllm_env --symlinks

# Activate environment
source vllm_env/bin/activate

# Install vLLM with TPU support
echo "Installing vllm-tpu..."
pip install vllm-tpu

echo
echo "=== Setup complete on $(hostname) ==="
echo "Python: $(which python3.12)"
echo "vLLM-TPU installed: $(pip show vllm-tpu 2>/dev/null | grep Version || pip show vllm 2>/dev/null | grep Version || echo 'not found')"
