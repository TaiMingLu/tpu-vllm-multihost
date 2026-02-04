# Model Architecture Investigation: llama3.1-05b/1b/3b

## Problem Summary

When attempting to use the converted MaxText checkpoints with vLLM for inference, we encountered a shape mismatch error:

```
TypeError: cannot reshape array of shape (1024, 2048) into shape (8, 64, 2048)
```

This document explains the root cause and solution.

---

## Key Discovery: Non-Standard Expansion Architecture

### What We Found

The custom llama3.1-05b/1b/3b models use a **non-standard expansion architecture** where `head_dim=128` is larger than what the standard formula `hidden_size / num_attention_heads` would calculate.

### Example: llama3.1-1b Architecture

**Configured dimensions:**
- `hidden_size`: 2048
- `num_attention_heads`: 32
- `num_key_value_heads`: 8
- `head_dim`: **128** (from MaxText config)

**Standard Llama would have:**
- `head_dim = hidden_size / num_attention_heads = 2048 / 32 = 64`

**Actual trained model has:**
- `head_dim = 128` (explicitly configured in MaxText)

### Actual Weight Shapes in Checkpoint

From inspecting the MaxText Orbax checkpoint at `checkpoint_24999/0/items`:

```python
# Attention weights are stored as 4D tensors:
Query:  (2048, 16, 32, 128)  # [hidden_size, sharding_dim, num_query_heads, head_dim]
Key:    (2048, 16, 8,  128)  # [hidden_size, sharding_dim, num_kv_heads, head_dim]
Value:  (2048, 16, 8,  128)  # [hidden_size, sharding_dim, num_kv_heads, head_dim]
Output: (32, 16, 128, 2048)  # [num_query_heads, sharding_dim, head_dim, hidden_size]

# Where sharding_dim=16 is for tensor parallelism
```

### Architecture Flow

```
Input: [batch, seq_len, 2048]
   ↓
Query Projection:  2048 → 32 × 128 = 4096  (2x expansion!)
Key Projection:    2048 → 8 × 128 = 1024
Value Projection:  2048 → 8 × 128 = 1024
   ↓
Attention: Q × K^T × V
   ↓
Output shape: [batch, seq_len, 32, 128] = [batch, seq_len, 4096]
   ↓
Output Projection: 4096 → 2048  (contracts back)
   ↓
Output: [batch, seq_len, 2048]
```

This is an **expansion-contraction** architecture where attention operates in a higher-dimensional space (4096) than the hidden dimension (2048).

---

## Why the Error Occurred

### vLLM's Expectation

When vLLM loads a HuggingFace Llama model with:
```python
config = LlamaConfig(
    hidden_size=2048,
    num_attention_heads=32,
    num_key_value_heads=8,
)
```

It **calculates** `head_dim = hidden_size / num_attention_heads = 2048 / 32 = 64`

### Weight Reshaping Logic

vLLM tries to reshape the K projection weight:
```python
# Loaded weight from HF: (1024, 2048)
# vLLM tries to reshape to: (num_kv_heads, head_dim, hidden_size)
weight.reshape(8, 64, 2048)  # ❌ FAILS!

# Because 1024 × 2048 = 2,097,152 elements
# But 8 × 64 × 2048 = 1,048,576 elements (half!)
```

### Why the Mismatch?

The actual weight should be reshaped to:
```python
weight.reshape(8, 128, 2048)  # ✓ Correct with head_dim=128
# 8 × 128 × 2048 = 2,097,152 elements (matches!)
```

---

## How MaxText Config Works

### MaxText Training Configuration

In `MaxText/configs/models/llama3.1-1b.yml`:
```yaml
base_emb_dim: 2048
base_num_query_heads: 32
base_num_kv_heads: 8
base_mlp_dim: 8192
head_dim: 128  # ← Explicitly specified!
```

### MaxText Attention Layer Implementation

From `MaxText/layers/attentions.py`:

```python
class Attention(nn.Module):
    config: Config
    num_query_heads: int
    num_kv_heads: int
    head_dim: int  # ← Used directly from config

    def query_projection(self, inputs_q):
        # Creates projection with output shape: (num_query_heads, head_dim)
        query_proj = dense_general(
            inputs_shape=inputs_q.shape,
            out_features_shape=(self.num_query_heads, self.head_dim),  # (32, 128)
            ...
        )(inputs_q)
        return query_proj
```

**Key insight:** MaxText uses the explicit `head_dim` value from config, it does NOT calculate it from `base_emb_dim / num_heads`.

This means MaxText can train models with arbitrary head dimensions, including expansion architectures where `head_dim > hidden_size / num_heads`.

---

## The Solution

### Fix 1: Keep Correct `dims_per_head` in MODEL_PARAMS_DICT

In `MaxText/llama_or_mistral_ckpt.py`:

```python
MODEL_PARAMS_DICT = {
    "llama3.1-05b": {
        "base_emb_dim": 1024,
        "num_heads": 32,
        "num_kv_heads": 8,
        "dims_per_head": 128,  # ✓ Keep at 128 (not 32)
        # ...
    },
    "llama3.1-1b": {
        "base_emb_dim": 2048,
        "num_heads": 32,
        "num_kv_heads": 8,
        "dims_per_head": 128,  # ✓ Keep at 128 (not 64)
        # ...
    },
    "llama3.1-3b": {
        "base_emb_dim": 3072,
        "num_heads": 32,
        "num_kv_heads": 8,
        "dims_per_head": 128,  # ✓ Keep at 128 (not 96)
        # ...
    },
}
```

### Fix 2: Add Explicit `head_dim` to HuggingFace Config

In `MaxText/llama_mistral_mixtral_orbax_to_hf.py`:

```python
elif model_size == "llama3.1-1b":
    from transformers import LlamaConfig
    config = LlamaConfig(
        hidden_size=2048,
        intermediate_size=8192,
        num_attention_heads=32,
        num_hidden_layers=16,
        num_key_value_heads=8,
        vocab_size=128256,
        rms_norm_eps=1e-5,
        rope_theta=500000,
        head_dim=128,  # ← ADD THIS! Overrides default calculation
    )
    model = AutoModelForCausalLM.from_config(config)
```

The `head_dim=128` parameter tells HuggingFace (and by extension, vLLM) to use the explicit value instead of calculating `hidden_size / num_attention_heads`.

---

## Model Dimensions Comparison

### Standard Llama 1B (hypothetical)

```
hidden_size: 2048
num_attention_heads: 32
head_dim: 64 (calculated)

Query weight:  [2048, 32 × 64]  = [2048, 2048] = 4.2M params
Key weight:    [2048, 8 × 64]   = [2048, 512]  = 1.0M params
Value weight:  [2048, 8 × 64]   = [2048, 512]  = 1.0M params
Output weight: [32 × 64, 2048]  = [2048, 2048] = 4.2M params
Total attention: ~10.4M params
```

### Your Custom llama3.1-1b

```
hidden_size: 2048
num_attention_heads: 32
head_dim: 128 (explicit)

Query weight:  [2048, 32 × 128] = [2048, 4096] = 8.4M params
Key weight:    [2048, 8 × 128]  = [2048, 1024] = 2.1M params
Value weight:  [2048, 8 × 128]  = [2048, 1024] = 2.1M params
Output weight: [32 × 128, 2048] = [4096, 2048] = 8.4M params
Total attention: ~21M params (2x more!)
```

Your model has **2x more attention parameters** than a standard Llama 1B would have.

---

## All Three Custom Models

| Model | hidden_size | num_heads | Standard head_dim | Actual head_dim | Expansion Factor |
|-------|-------------|-----------|-------------------|-----------------|------------------|
| llama3.1-05b | 1024 | 32 | 32 | **128** | 4x |
| llama3.1-1b | 2048 | 32 | 64 | **128** | 2x |
| llama3.1-3b | 3072 | 32 | 96 | **128** | 1.33x |

All three models use `head_dim=128`, creating increasingly less expansion as the hidden size grows.

---

## Implementation Status

### Completed
- ✅ Identified the architectural difference
- ✅ Reverted incorrect dims_per_head values in llama_or_mistral_ckpt.py (back to 128)
- ✅ Added explicit head_dim=128 to HuggingFace configs in llama_mistral_mixtral_orbax_to_hf.py
- ✅ Committed and pushed fixes to MaxText fork

### To Do
- [ ] Reconvert llama3.1-1b checkpoint with correct config
- [ ] Test generation with vLLM
- [ ] Convert llama3.1-05b checkpoint
- [ ] Convert llama3.1-3b checkpoint

---

## Lessons Learned

1. **MaxText allows non-standard architectures**: The `head_dim` parameter can be set independently of `hidden_size / num_heads`, enabling expansion architectures.

2. **HuggingFace supports explicit head_dim**: The `LlamaConfig` accepts a `head_dim` parameter that overrides the default calculation.

3. **Always inspect actual checkpoint shapes**: Don't assume standard architectures - verify weight shapes directly from checkpoints when debugging conversion issues.

4. **Config values may not match standard formulas**: When a framework allows explicit configuration, always check if values are calculated or explicitly set.

---

## References

- MaxText training config: `maxtext_distillation/MaxText/configs/models/llama3.1-1b.yml`
- Attention implementation: `maxtext_distillation/MaxText/layers/attentions.py:1348`
- Conversion script: `maxtext_distillation/MaxText/llama_mistral_mixtral_orbax_to_hf.py`
- Model params dict: `maxtext_distillation/MaxText/llama_or_mistral_ckpt.py:186`

---

**Date:** 2024-12-02
**Investigation by:** Claude Code
