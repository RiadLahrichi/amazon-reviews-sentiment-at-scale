"""Compare three classic classifiers in Spark on the full dataset.

    python compare_models.py
    python compare_models.py --size 500000     (smaller and faster)

The TF-IDF vectors are computed once, then Naive Bayes, logistic regression
and a linear SVM are all trained on the same vectors.
Writes results/compare_models.md.
"""

import argparse
import time

import pandas as pd
from pyspark.ml.classification import LinearSVC, NaiveBayes
from sklearn.metrics import accuracy_score, f1_score

from common import (add_class_weights, add_terms, fit_tfidf, get_spark, load_sample_spark,
                    spark_logistic_regression)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", default="all", help='number of reviews, or "all"')
    args = parser.parse_args()

    spark = get_spark()
    train, test = load_sample_spark(spark, args.size)
    train, train_rows = add_class_weights(add_terms(train))
    test = add_terms(test)

    # Turn the text into TF-IDF vectors once. cache() keeps the vectors in
    # memory, because all three models are trained on them.
    tfidf = fit_tfidf(train)
    train_vectors = tfidf.transform(train).select("features", "label", "weight").cache()
    test_vectors = tfidf.transform(test).select("features", "label").cache()
    # Spark is lazy: nothing is computed until a result is needed. count()
    # forces the vectors to be built now, so the first model's training time
    # doesn't include building them.
    test_rows = test_vectors.count()
    train_vectors.count()

    models = {
        "Naive Bayes": NaiveBayes(weightCol="weight"),
        "Logistic regression": spark_logistic_regression(train_rows),
        # the same penalty and settings as the logistic regression, so the comparison is fair
        "Linear SVM": LinearSVC(weightCol="weight", regParam=1.0 / train_rows,
                                standardization=False),
    }

    rows = []
    for name, model in models.items():
        print(f"training {name}...")
        start = time.time()
        fitted = model.fit(train_vectors)
        seconds = time.time() - start

        predictions = fitted.transform(test_vectors).select("label", "prediction").toPandas()
        y_true = predictions["label"]
        y_pred = predictions["prediction"].astype(int)
        f1_negative, f1_positive = f1_score(y_true, y_pred, average=None)
        rows.append({
            "model": name,
            "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 3),
            "f1_negative": round(f1_negative, 3),
            "f1_positive": round(f1_positive, 3),
            "accuracy": round(accuracy_score(y_true, y_pred), 3),
            "training_seconds": round(seconds),
        })

    table = pd.DataFrame(rows).to_markdown(index=False)
    summary = f"Trained on {train_rows:,} reviews, tested on {test_rows:,} reviews.\n\n{table}\n"
    with open("results/compare_models.md", "w") as f:
        f.write(summary)
    print(summary)


if __name__ == "__main__":
    main()
