# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is `tpu-inference`, a hardware plugin for vLLM that provides high-performance LLM inference on Google TPUs. It unifies JAX and PyTorch model execution under a single backend, allowing PyTorch models to run efficiently on TPU without code changes while also supporting native JAX implementations.

**Key Design Principles:**
- Unified backend: JAX and PyTorch models share the same lowering path
- Framework flexibility: Run PyTorch model definitions on TPU via torchax, or use native JAX implementations
- vLLM standardization: Maintains the same user experience, telemetry, and interface as vLLM

**Compatible TPU Generations:**
- Recommended: v5e, v6e
- Experimental: v3, v4, v5p

## Architecture

The codebase follows a layered architecture with framework-specific and common components:

### Directory Structure

**Core Components:**
- `tpu_inference/core/`: Core TPU execution engine
  - `core_tpu.py`: Main TPU execution logic, disaggregated execution support
  - `sched/`: Scheduling logic for request handling
- `tpu_inference/runner/`: Model execution orchestration
  - `tpu_runner.py`: Main runner coordinating model execution (~1700 lines, primary entry point)
  - `compilation_manager.py`: JAX compilation and caching
  - `kv_cache_manager.py`: KV cache management
  - `input_batch.py`: Input batch handling
- `tpu_inference/worker/`: Worker processes for distributed execution
- `tpu_inference/executors/`: Distributed execution via Ray

**Model and Layer Implementations:**
- `tpu_inference/models/`:
  - `common/model_loader.py`: **Model registry** - register new models here
  - `jax/`: Native JAX model implementations (llama3, llama4, qwen2/3, deepseek_v3, etc.)
  - `vllm/`: PyTorch model wrappers (via torchax)
- `tpu_inference/layers/`:
  - `common/`: Framework-agnostic layers
  - `jax/`: JAX-specific layers (attention, mlp, sampling)
  - `vllm/`: vLLM/PyTorch-specific layers

**Kernels:**
- `tpu_inference/kernels/`: High-performance TPU kernels (Pallas-based)
  - `flash_attention/`: Flash attention implementations
  - `ragged_paged_attention/`: Paged attention for variable-length sequences (v2/, v3/ for TPU generations)
  - `fused_moe/`: Fused mixture-of-experts kernels
  - `quantized_matmul/`: Quantization kernels
  - `mla/`: Multi-head latent attention (DeepSeek)
  - `collectives/`: Distributed communication primitives

**Configuration and Utilities:**
- `tpu_inference/envs.py`: **CRITICAL** - Centralized environment variable management
- `tpu_inference/env_override.py`: Environment variable overrides (imported first)
- `tpu_inference/tpu_info.py`: TPU hardware information utilities
- `tpu_inference/platforms/`: Platform detection and configuration

### Model Implementation Types

The codebase supports two model implementation strategies:

1. **JAX Native Models** (`tpu_inference/models/jax/`):
   - Written in JAX/Flax NNX for maximum TPU performance
   - Examples: llama3, llama4, qwen2, qwen3, deepseek_v3, llama_guard_4
   - Require manual model definition but offer best performance

2. **vLLM Native Models** (`tpu_inference/models/vllm/`):
   - Use PyTorch model definitions from vLLM upstream
   - Converted to JAX via torchax at runtime
   - Broader model support with minimal code changes
   - Managed by `vllm_model_wrapper.py`

## Development Commands

### Installation

```bash
# Standard installation
pip install -e .

# For TPU v7x support
pip install -r requirements_v7x.txt
```

### Testing

```bash
# All tests
pytest

# By category
pytest tests/core/      # Core execution logic
pytest tests/models/    # Model implementations
pytest tests/layers/    # Layers and components
pytest tests/kernels/   # Custom kernels
pytest tests/e2e/       # End-to-end tests

# Single test
pytest tests/test_envs.py::test_env_variable_name

# Quick testing (skip JAX warmup)
SKIP_JAX_PRECOMPILE=1 pytest tests/test_envs.py
```

### Linting and Formatting

```bash
# Install pre-commit hooks (required)
pip install pre-commit
pre-commit install --hook-type pre-commit --hook-type commit-msg

# Run manually
pre-commit run --all-files
```

Tools: `yapf` (Python), `isort` (imports), `ruff` (linting), `clang-format` (C++/CUDA)

### Running Models

```bash
# Basic inference
python examples/offline_inference.py --model meta-llama/Llama-3.2-1B-Instruct

# Multi-modal
python examples/multi_modal_inference.py --model <MODEL_NAME>

# LoRA
python examples/offline_lora_inference.py --model <BASE_MODEL> --lora <LORA_PATH>
```

```python
# Using vLLM API
from vllm import LLM
llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct")
outputs = llm.generate(["Hello, my name is"], sampling_params)
```

## Multi-Host Execution

For TPU pods spanning multiple hosts:

```bash
# One-time setup (install vLLM on all workers)
./setup_multihost.sh <tpu-name>

# Run generation
./run_multihost.sh <tpu-name> <num-chips>
# Example: ./run_multihost.sh my-v6e-32 32

# Or direct usage
python3 multihost_runner.py \
  --tpu-name=${TPU_NAME} \
  --command="export TENSOR_PARALLEL_SIZE=${TP_SIZE} && bash full_loop_vllm_v6e_multi.sh" \
  --script-dir=.
```

**Tensor Parallel Size** = total chips across all hosts (v5e-16→16, v6e-32→32)

Worker logs: `/tmp/multihost_worker_N.log`

## Environment Variables

**CRITICAL:** Always access environment variables through `tpu_inference.envs`:

```python
# Correct
from tpu_inference import envs
if envs.SKIP_JAX_PRECOMPILE:
    ...

# Incorrect (breaks centralized management)
import os
if os.getenv("SKIP_JAX_PRECOMPILE"):
    ...
```

**Key Variables:**
- `JAX_PLATFORMS`: Platform selection ("tpu", "cpu", "proxy")
- `TPU_ACCELERATOR_TYPE`: TPU type (e.g., "v5litepod-16", "v6e-8")
- `TPU_NAME`, `TPU_WORKER_ID`: TPU resource identification
- `SKIP_JAX_PRECOMPILE`: Skip JAX precompilation (quick tests)
- `VLLM_XLA_CHECK_RECOMPILATION`: Check for unexpected recompilations
- `PREFILL_SLICES`, `DECODE_SLICES`: Disaggregated serving config

## CI/CD

Buildkite pipelines in `.buildkite/`:
- `pipeline_jax.yml`: JAX model testing
- `pipeline_torch.yml`: PyTorch model testing
- `main.yml`: Main CI pipeline

**Adding Models to CI:**
```bash
cd .buildkite/pipeline_generation

# TPU-optimized
python add_model_to_ci.py --model-name meta-llama/Llama-3.1-8B --queue tpu_v6e_queue

# vLLM-native
python add_model_to_ci.py --model-name <MODEL_NAME> --queue tpu_v6e_queue --type vllm-native
```

## Code Organization Patterns

### Framework Segregation

- JAX-only code: `*/jax/` subdirectories
- vLLM/PyTorch-only code: `*/vllm/` subdirectories
- Shared code: `*/common/` subdirectories

### Adding a New JAX Model

1. Create model file: `tpu_inference/models/jax/<model_name>.py`
   - Define `class NewModel(nnx.Module)` with constructor taking `(VllmConfig, nnx.Rngs, jax.sharding.Mesh)`
   - Implement `__call__` for forward pass
   - Implement `compute_logits` for logit generation
   - Implement `load_weights` for HuggingFace weight loading
2. Register in `tpu_inference/models/common/model_loader.py`
3. Add layers to `tpu_inference/layers/jax/` if needed
4. Add tests: `tests/models/jax/test_<model_name>.py`
5. Add to CI: `python add_model_to_ci.py --model-name ...`

### Adding a vLLM Model

1. Ensure PyTorch model exists in vLLM upstream
2. Test via `vllm_model_wrapper.py`
3. Add tests: `tests/models/vllm/`
4. Add to CI: `python add_model_to_ci.py --type vllm-native ...`

## Implementation Details

### JAX Compilation
- JIT compilation with caching in `compilation_manager.py`
- First run triggers compilation (slow), subsequent runs fast
- Use `SKIP_JAX_PRECOMPILE=1` to skip warmup
- Debug recompilation with `VLLM_XLA_CHECK_RECOMPILATION=1`

### Multi-host Coordination
- All hosts execute the same compiled code
- Worker 0 coordinates I/O operations
- Communication via JAX collectives (all-reduce, all-gather)
- Internal IPs for TPU-to-TPU communication

### KV Cache
- Paged attention with block-based cache in `kv_cache_manager.py`
- Dynamic block allocation by sequence length

### Quantization
- Kernels in `kernels/quantized_matmul/`
- Qwix integration for loading pre-quantized models
- Configs in `models/jax/utils/quantization/configs/`

## External Resources

- [Documentation](https://docs.vllm.ai/projects/tpu/en/latest/)
- [vLLM Project](https://github.com/vllm-project/vllm)
- [Developer Slack](https://slack.vllm.ai) (#sig-tpu channel)
- [User Forum](https://discuss.vllm.ai/c/hardware-support/google-tpu-support/27)
- [TPU Documentation](https://cloud.google.com/tpu/docs)
- [JAX Documentation](https://jax.readthedocs.io)
