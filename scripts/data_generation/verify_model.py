"""Verify that a model is in the correct HuggingFace format for vLLM.

Usage:
  python scripts/data_generation/verify_model.py --model-path /path/to/model
"""

import argparse
import json
import os
import sys


def check_file_exists(path, filename):
    """Check if a file exists and print status."""
    filepath = os.path.join(path, filename)
    exists = os.path.exists(filepath)
    status = "✓" if exists else "✗"
    print(f"  {status} {filename}")
    return exists


def verify_model(model_path):
    """Verify model directory has required files."""
    print(f"\nVerifying model at: {model_path}\n")

    if not os.path.exists(model_path):
        print(f"ERROR: Model path does not exist: {model_path}")
        return False

    if not os.path.isdir(model_path):
        print(f"ERROR: Model path is not a directory: {model_path}")
        return False

    # Check for required files
    print("Checking required files:")
    has_config = check_file_exists(model_path, "config.json")
    has_tokenizer_config = check_file_exists(model_path, "tokenizer_config.json")
    has_tokenizer_json = check_file_exists(model_path, "tokenizer.json")

    # Check for model weights (various formats)
    print("\nChecking model weights:")
    has_safetensors = any(
        f.endswith(".safetensors") for f in os.listdir(model_path)
    )
    has_pytorch = any(
        f.endswith(".bin") or f == "pytorch_model.bin" for f in os.listdir(model_path)
    )

    if has_safetensors:
        print("  ✓ Found .safetensors weights")
    elif has_pytorch:
        print("  ✓ Found .bin weights")
    else:
        print("  ✗ No model weights found (.safetensors or .bin)")

    # Read and display config
    if has_config:
        print("\nModel configuration:")
        try:
            with open(os.path.join(model_path, "config.json")) as f:
                config = json.load(f)
                print(f"  Model type: {config.get('model_type', 'unknown')}")
                print(f"  Architecture: {config.get('architectures', ['unknown'])[0]}")
                print(f"  Hidden size: {config.get('hidden_size', 'unknown')}")
                print(f"  Num layers: {config.get('num_hidden_layers', 'unknown')}")
                print(f"  Num attention heads: {config.get('num_attention_heads', 'unknown')}")
                print(f"  Vocab size: {config.get('vocab_size', 'unknown')}")
        except Exception as e:
            print(f"  ERROR reading config.json: {e}")

    # Summary
    print("\nVerification Summary:")
    all_required = has_config and (has_safetensors or has_pytorch)
    tokenizer_ok = has_tokenizer_config or has_tokenizer_json

    if all_required and tokenizer_ok:
        print("  ✓ Model appears to be in valid HuggingFace format")
        print("  ✓ Ready for vLLM")
        return True
    else:
        print("  ✗ Model is missing required files")
        if not has_config:
            print("    - Missing config.json")
        if not (has_safetensors or has_pytorch):
            print("    - Missing model weights (.safetensors or .bin)")
        if not tokenizer_ok:
            print("    - Missing tokenizer files (tokenizer_config.json or tokenizer.json)")
        print("\n  You may need to convert your model to HuggingFace format.")
        print("  See scripts/data_generation/README.md for instructions.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Verify model is in HuggingFace format for vLLM"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model directory"
    )
    args = parser.parse_args()

    success = verify_model(args.model_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
