"""Convert MaxText checkpoint to HuggingFace format for custom model sizes.

This script helps convert custom MaxText models (1b, 3b, 4b-width) to HuggingFace format
by creating the appropriate config.json and using MaxText's conversion script.

Usage:
  python convert_maxtext_to_hf.py \
    --model-size llama3.1-1b \
    --maxtext-checkpoint gs://bucket/ckpts/param_only/checkpoint_24999 \
    --output-path /local/path/to/hf/model
"""

import argparse
import json
import os
import subprocess
import sys


# Model size configurations mapping MaxText configs to HuggingFace
MODEL_CONFIGS = {
    "llama3.1-05b": {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 1024,
        "intermediate_size": 8192,
        "num_attention_heads": 32,
        "num_hidden_layers": 12,
        "num_key_value_heads": 8,
        "vocab_size": 128256,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000,
        "model_type": "llama",
        "torch_dtype": "bfloat16",
        "transformers_version": "4.40.0",
    },
    "llama3.1-1b": {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "num_attention_heads": 32,
        "num_hidden_layers": 16,
        "num_key_value_heads": 8,
        "vocab_size": 128256,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000,
        "model_type": "llama",
        "torch_dtype": "bfloat16",
        "transformers_version": "4.40.0",
    },
    "llama3.1-3b": {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 3072,
        "intermediate_size": 8192,
        "num_attention_heads": 32,
        "num_hidden_layers": 28,
        "num_key_value_heads": 8,
        "vocab_size": 128256,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000,
        "model_type": "llama",
        "torch_dtype": "bfloat16",
        "transformers_version": "4.40.0",
    },
}


def create_hf_config(model_size, output_path):
    """Create HuggingFace config.json for the model."""
    if model_size not in MODEL_CONFIGS:
        print(f"Error: Unknown model size '{model_size}'")
        print(f"Available: {list(MODEL_CONFIGS.keys())}")
        sys.exit(1)

    config = MODEL_CONFIGS[model_size]

    os.makedirs(output_path, exist_ok=True)
    config_path = os.path.join(output_path, "config.json")

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Created config.json at {config_path}")
    return config_path


def run_maxtext_conversion(maxtext_checkpoint, model_size, output_path, maxtext_dir):
    """Run MaxText's conversion script using the ssh command method."""

    # The checkpoint path should point to the items directory
    if not maxtext_checkpoint.endswith("/items"):
        checkpoint_path = os.path.join(maxtext_checkpoint, "0/items")
    else:
        checkpoint_path = maxtext_checkpoint

    print(f"\nConverting MaxText checkpoint to HuggingFace format...")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Model size: {model_size}")
    print(f"  Output: {output_path}")

    # Create a wrapper script that will be executed
    conversion_cmd = f"""
cd {maxtext_dir}
source ~/maxtext_env/bin/activate

# Set single-host environment
export PJRT_DEVICE=TPU
unset JAX_COORDINATOR_ADDRESS
export JAX_PROCESS_COUNT=1
export JAX_LOCAL_DEVICE_COUNT=8

python3 -m MaxText.llama_mistral_mixtral_orbax_to_hf \\
  MaxText/configs/base.yml \\
  load_parameters_path={checkpoint_path} \\
  model_name=llama3.1-8b \\
  hardware=gpu \\
  hf_model_path={output_path}_temp
"""

    print("\nInstructions:")
    print("=" * 70)
    print("Run the following command on your TPU VM:")
    print("=" * 70)
    print(conversion_cmd)
    print("=" * 70)

    print(f"\nAfter conversion completes:")
    print(f"1. The weights will be in: {output_path}_temp")
    print(f"2. Copy the config.json we created: cp {output_path}/config.json {output_path}_temp/")
    print(f"3. Rename: mv {output_path}_temp {output_path}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert MaxText checkpoint to HuggingFace format"
    )
    parser.add_argument(
        "--model-size",
        type=str,
        required=True,
        choices=list(MODEL_CONFIGS.keys()),
        help="Model size (llama3.1-1b, llama3.1-3b, llama3.1-4b-width)"
    )
    parser.add_argument(
        "--maxtext-checkpoint",
        type=str,
        required=True,
        help="Path to MaxText param-only checkpoint (e.g., gs://bucket/ckpts/checkpoint_24999)"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Output path for HuggingFace model"
    )
    parser.add_argument(
        "--maxtext-dir",
        type=str,
        default="~/maxtext",
        help="MaxText directory on TPU VM (default: ~/maxtext)"
    )

    args = parser.parse_args()

    # Step 1: Create HuggingFace config
    print("Step 1: Creating HuggingFace config.json...")
    create_hf_config(args.model_size, args.output_path)

    # Step 2: Show instructions for MaxText conversion
    print("\nStep 2: Convert weights using MaxText...")
    run_maxtext_conversion(
        args.maxtext_checkpoint,
        args.model_size,
        args.output_path,
        args.maxtext_dir
    )

    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print(f"After running the conversion command on your TPU VM:")
    print(f"1. Verify the model with:")
    print(f"   python scripts/data_generation/verify_model.py --model-path {args.output_path}")
    print(f"2. Use the model for data generation:")
    print(f"   export MODEL_PATH={args.output_path}")
    print(f"   bash scripts/data_generation/run_sequence_kd.sh")


if __name__ == "__main__":
    main()
