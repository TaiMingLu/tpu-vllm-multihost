#!/usr/bin/env python3
"""Basic inference example for vLLM on TPU."""

import argparse
from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=2048)
    args = parser.parse_args()

    prompts = [
        "The capital of France is",
        "Once upon a time",
        "In machine learning,",
    ]

    print(f"Loading model: {args.model}")
    print(f"Tensor parallel size: {args.tensor_parallel_size}")

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
    )

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=50,
    )

    print("\nGenerating responses...\n")
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print(f"Prompt: {prompt}")
        print(f"Generated: {output.outputs[0].text}")
        print("-" * 80)


if __name__ == "__main__":
    main()
