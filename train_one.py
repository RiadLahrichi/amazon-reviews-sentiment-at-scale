"""Train one engine on one sample size and add the result to results/benchmark.csv.

    python train_one.py --engine sklearn --size 100000
    python train_one.py --engine spark --size all

run_benchmark.sh calls this once for every engine and size. Each call is a
new Python process, so the memory used by one run can't affect the next.
If a run uses more memory than the machine has, Linux stops the process and
run_benchmark.sh records it as "crashed or ran out of memory".

Both engines get the same reviews, the same features (words and word pairs,
hashed into 262,144 buckets, then TF-IDF) and the same model (logistic
regression with balanced class weights). Only the engine is different.
"""

import argparse
import os
import resource
import time

import pandas as pd
import psutil
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline

from common import (NUM_FEATURES, add_class_weights, add_terms, fit_tfidf, get_spark,
                    load_sample_pandas, load_sample_spark, spark_logistic_regression)

RESULTS = "results/benchmark.csv"


def train_sklearn(size):
    train, test = load_sample_pandas(size)
    model = make_pipeline(
        # alternate_sign=False keeps plain word counts, like Spark's HashingTF.
        # token_pattern splits on spaces like Spark's Tokenizer. The default
        # would drop one-letter words, which Spark keeps.
        HashingVectorizer(n_features=NUM_FEATURES, ngram_range=(1, 2), alternate_sign=False,
                          token_pattern=r"\S+"),
        TfidfTransformer(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    model.fit(train["clean_text"], train["label"])
    predictions = model.predict(test["clean_text"])
    return len(train), test["label"], predictions


def train_spark(size):
    spark = get_spark()
    train, test = load_sample_spark(spark, size)
    train, train_rows = add_class_weights(add_terms(train))
    test = add_terms(test)

    tfidf = fit_tfidf(train)
    # see common.py for the settings that make it match scikit-learn
    model = spark_logistic_regression(train_rows).fit(tfidf.transform(train))
    predictions = model.transform(tfidf.transform(test)).select("label", "prediction").toPandas()
    # Spark is not stopped here: it stops when this script ends. Until then
    # its Java process is still running, so peak_memory_gb() can measure it.
    return train_rows, predictions["label"], predictions["prediction"].astype(int)


def peak_memory_gb():
    """Highest memory used by this Python process, plus the memory of its child
    processes. Spark does its work in a separate Java process started by
    PySpark, so that's where Spark's memory shows up. Java rarely gives memory
    back to the system, so its current size is close to its peak."""
    python_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024  # Linux reports KB
    children = sum(child.memory_info().rss for child in psutil.Process().children(recursive=True))
    return (python_peak + children) / 1024 ** 3


def save_result(row):
    os.makedirs("results", exist_ok=True)
    write_header = not os.path.exists(RESULTS)
    pd.DataFrame([row]).to_csv(RESULTS, mode="a", header=write_header, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["sklearn", "spark"], required=True)
    parser.add_argument("--size", required=True, help='number of reviews, or "all"')
    args = parser.parse_args()

    row = {"engine": args.engine, "size": args.size, "status": "ok",
           "train_rows": None, "seconds": None, "memory_gb": None, "macro_f1": None}

    start = time.time()
    try:
        if args.engine == "sklearn":
            train_rows, y_true, y_pred = train_sklearn(args.size)
        else:
            train_rows, y_true, y_pred = train_spark(args.size)
        row["seconds"] = round(time.time() - start, 1)
        row["train_rows"] = train_rows
        row["macro_f1"] = round(f1_score(y_true, y_pred, average="macro"), 4)
    except MemoryError:
        row["status"] = "out of memory"
    except Exception as error:
        # When Spark's Java process runs out of memory, Python gets a general error
        if "OutOfMemoryError" in str(error):
            row["status"] = "out of memory"
        else:
            row["status"] = f"failed: {type(error).__name__}"

    row["memory_gb"] = round(peak_memory_gb(), 2)
    save_result(row)
    print(row)


if __name__ == "__main__":
    main()
