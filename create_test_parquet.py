#!/usr/bin/env python3
"""
Create a test parquet file with sample prompts for testing the generation pipeline.
"""

import argparse
import pandas as pd
from pathlib import Path


def create_test_data(num_samples: int = 10) -> pd.DataFrame:
    """Create sample prompts for testing."""

    prompts = [
        "Write a short story about a robot learning to paint.",
        "Explain quantum computing in simple terms.",
        "What are the benefits of regular exercise?",
        "Describe a futuristic city in the year 2150.",
        "How does photosynthesis work?",
        "Write a haiku about autumn.",
        "What is the theory of relativity?",
        "Describe the perfect pizza.",
        "How can I improve my public speaking skills?",
        "What makes a good leader?",
        "Explain the water cycle.",
        "Write a dialogue between a detective and a suspect.",
        "What are the main causes of climate change?",
        "Describe your favorite book and why you like it.",
        "How does machine learning work?",
        "Write a recipe for chocolate chip cookies.",
        "What are the benefits of meditation?",
        "Describe a memorable travel experience.",
        "How do airplanes fly?",
        "What is the meaning of life?",
    ]

    # Take the requested number of samples
    selected_prompts = prompts[:num_samples]

    # Create DataFrame
    df = pd.DataFrame({
        'prompt': selected_prompts,
        'id': range(len(selected_prompts)),
        'category': ['test'] * len(selected_prompts)
    })

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Create test parquet file with sample prompts"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="test_prompts.parquet",
        help="Output parquet file path"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of sample prompts to generate"
    )

    args = parser.parse_args()

    # Create test data
    df = create_test_data(args.num_samples)

    # Save to parquet
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(args.output, engine='pyarrow', compression='snappy', index=False)

    print(f"✓ Created test parquet file: {args.output}")
    print(f"  Number of samples: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  File size: {output_path.stat().st_size / 1024:.2f} KB")

    # Show first few samples
    print("\nFirst 3 prompts:")
    for i, row in df.head(3).iterrows():
        print(f"  {i+1}. {row['prompt']}")


if __name__ == "__main__":
    main()
