"""Step 1: raw review file -> cleaned, labelled Parquet.

    python prepare_data.py data/raw/Video_Games.jsonl.gz
    python prepare_data.py data/raw/Video_Games.jsonl.gz --limit 50000   (quick test)

Drops empty and duplicate reviews, cleans the text, labels by stars
(4-5 = 1, 1-2 = 0, 3 = none) and saves data/reviews.parquet.
Parquet because a .gz can only be read by one core.
"""

import argparse
import json
import os

from pyspark.ml.feature import RegexTokenizer, StopWordsRemover
from pyspark.sql import functions as F

from cleaning import STOP_WORDS, clean_text
from common import DATA, STATS, get_spark

# giving the schema up front saves Spark a full pass over the file to guess it
SCHEMA = ("rating DOUBLE, text STRING, parent_asin STRING, user_id STRING, "
          "timestamp LONG, helpful_vote LONG, verified_purchase BOOLEAN")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="path to the .jsonl.gz file")
    parser.add_argument("--limit", type=int, help="only use the first N reviews")
    args = parser.parse_args()

    spark = get_spark()
    reviews = spark.read.schema(SCHEMA).json(args.input)
    if args.limit:
        reviews = reviews.limit(args.limit)

    # a .gz is read as one partition, so split it up for all cores to share
    reviews = reviews.repartition(32)

    reviews = reviews.dropna(subset=["rating", "text"])
    reviews = reviews.dropDuplicates(["user_id", "parent_asin", "text"])

    # clean, split into words, drop stop words, join back
    reviews = reviews.withColumn("clean_text", clean_text(F.col("text")))
    reviews = RegexTokenizer(inputCol="clean_text", outputCol="all_words").transform(reviews)
    # locale set because some Linux defaults get rejected by Spark
    reviews = StopWordsRemover(inputCol="all_words", outputCol="words",
                               stopWords=STOP_WORDS, locale="en_US").transform(reviews)
    reviews = reviews.withColumn("clean_text", F.array_join("words", " "))
    reviews = reviews.withColumn("word_count", F.size("words"))
    reviews = reviews.filter(F.col("word_count") > 0)

    # 3 stars is mixed, so no label (they stay in the data for insights.py)
    reviews = reviews.withColumn(
        "label", F.when(F.col("rating") >= 4, 1).when(F.col("rating") <= 2, 0))

    # random numbers for sampling and the train/test split, see common.py
    reviews = reviews.withColumn("sample_key", F.rand(seed=42))
    reviews = reviews.withColumn("split_key", F.rand(seed=7))

    reviews = reviews.drop("all_words", "words", "user_id")
    reviews.write.mode("overwrite").parquet(DATA)
    save_stats(spark, args.input)


def save_stats(spark, source):
    saved = spark.read.parquet(DATA)
    counts = {row["label"]: row["count"] for row in saved.groupBy("label").count().collect()}
    stats = {
        "source": os.path.basename(source),
        "reviews": sum(counts.values()),
        "positive": counts.get(1, 0),
        "negative": counts.get(0, 0),
        "three_stars_no_label": counts.get(None, 0),
    }
    stats["labelled_rows"] = stats["positive"] + stats["negative"]
    stats["positive_share"] = round(stats["positive"] / stats["labelled_rows"], 3)

    os.makedirs("results", exist_ok=True)
    with open(STATS, "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
