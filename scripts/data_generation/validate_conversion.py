#!/usr/bin/env python3
"""
Quick validation script for converted HuggingFace models.
Tests that the model loads and can generate text.
"""

import argparse
from vllm import LLM, SamplingParams


def validate_model(model_path: str):
    """Validate that model loads and can generate text."""

    print(f"Loading model from: {model_path}")

    try:
        # Load model with vLLM
        llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            max_model_len=512,  # Small for quick test
        )
        print("✅ Model loaded successfully!")

        # Test generation
        test_prompts = [
            "The capital of France is",
            "Once upon a time",
            "To be or not to be",
        ]

        sampling_params = SamplingParams(
            temperature=0.0,  # Deterministic for testing
            top_p=1.0,
            max_tokens=20,
        )

        print("\nTesting text generation...")
        outputs = llm.generate(test_prompts, sampling_params)

        print("\n" + "="*80)
        print("Generation Examples:")
        print("="*80)

        for prompt, output in zip(test_prompts, outputs):
            generated_text = output.outputs[0].text
            print(f"\nPrompt: {prompt}")
            print(f"Generated: {generated_text}")

        print("\n" + "="*80)
        print("✅ Validation complete! Model is working correctly.")
        print("="*80)

        return True

    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Validate converted HuggingFace model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to converted HuggingFace model directory",
    )

    args = parser.parse_args()

    success = validate_model(args.model_path)
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
