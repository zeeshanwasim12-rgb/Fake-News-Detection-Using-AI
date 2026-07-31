"""Merge Kaggle's common Fake.csv and True.csv files into data/news.csv.

Usage:
    python data/prepare_kaggle_data.py --fake Fake.csv --real True.csv
"""

import argparse
from pathlib import Path

import pandas as pd


def main(fake_path, real_path, output_path):
    fake = pd.read_csv(fake_path)
    real = pd.read_csv(real_path)
    if "text" not in fake.columns or "text" not in real.columns:
        raise ValueError("Both input CSVs must contain a 'text' column.")

    fake = fake[["text"]].copy()
    real = real[["text"]].copy()
    fake["label"] = 1
    real["label"] = 0
    merged = pd.concat([fake, real], ignore_index=True).dropna(subset=["text"])
    merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"Saved {len(merged)} labeled articles to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Kaggle fake-news files for this project")
    parser.add_argument("--fake", required=True, help="Path to Fake.csv")
    parser.add_argument("--real", required=True, help="Path to True.csv")
    parser.add_argument("--output", default="data/news.csv", help="Destination text,label CSV")
    args = parser.parse_args()
    main(args.fake, args.real, args.output)
