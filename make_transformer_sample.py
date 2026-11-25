"""Save a sample of the labelled reviews for train_transformer.ipynb.

    python make_transformer_sample.py              (100,000 reviews)
    python make_transformer_sample.py --size 50000

Uses the same sampling and train/test split as the benchmark (see common.py),
and keeps the raw text for DistilBERT and the cleaned text for logistic
regression. Writes data/transformer_sample.parquet, small enough to upload to
Google Colab.
"""

import argparse
import os

import pandas as pd

from common import load_sample_pandas

OUTPUT = "data/transformer_sample.parquet"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", default="100000", help="number of labelled reviews")
    args = parser.parse_args()

    train, test = load_sample_pandas(args.size, columns=("text", "clean_text", "label"))
    train = train.assign(split="train")
    test = test.assign(split="test")
    sample = pd.concat([train, test]).drop(columns="split_key")

    sample.to_parquet(OUTPUT, index=False)
    size_mb = os.path.getsize(OUTPUT) / 1024 ** 2
    print(f"{len(train):,} training and {len(test):,} test reviews saved to {OUTPUT} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
