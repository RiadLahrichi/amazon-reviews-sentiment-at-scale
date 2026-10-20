"""Step 1: turn the raw review file into a clean Parquet dataset.

    python prepare_data.py data/raw/Video_Games.jsonl.gz
    python prepare_data.py data/raw/Video_Games.jsonl.gz --limit 50000   (quick test)

What it does:
  1. reads the raw reviews (one JSON object per line, gzipped)
  2. removes empty and duplicate reviews
  3. cleans the text and removes stop words
  4. labels each review: 4-5 stars = 1 (positive), 1-2 stars = 0 (negative),
     3 stars = no label
  5. saves the result to data/reviews.parquet, and a few counts to
     results/prepare_stats.json

Why Parquet: a .gz file can only be read from the start, so Spark reads it
with one CPU core. Parquet is stored as many files that can be read in
parallel, and a script can load only the columns it needs.
"""

import argparse
import json
import os

from pyspark.ml.feature import RegexTokenizer, StopWordsRemover
from pyspark.sql import functions as F

from cleaning import STOP_WORDS, clean_text
from common import DATA, STATS, get_spark

# The fields we use from each review. Telling Spark the types up front saves
# it from reading the whole file once just to guess them.
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

    # The .gz file comes in as a single partition, so only one core would
    # work on it. Split it into 32 pieces so all cores share the work.
    reviews = reviews.repartition(32)

    reviews = reviews.dropna(subset=["rating", "text"])
    reviews = reviews.dropDuplicates(["user_id", "parent_asin", "text"])

    # Clean the text, split it into words, remove stop words, join it back.
    reviews = reviews.withColumn("clean_text", clean_text(F.col("text")))
    reviews = RegexTokenizer(inputCol="clean_text", outputCol="all_words").transform(reviews)
    # locale is set because some Linux setups have a default that Spark rejects
    reviews = StopWordsRemover(inputCol="all_words", outputCol="words",
                               stopWords=STOP_WORDS, locale="en_US").transform(reviews)
    reviews = reviews.withColumn("clean_text", F.array_join("words", " "))
    reviews = reviews.withColumn("word_count", F.size("words"))
    reviews = reviews.filter(F.col("word_count") > 0)

    # 3-star reviews are mixed ("fine, nothing special"), so they get no
    # label. They stay in the data for insights.py.
    reviews = reviews.withColumn(
        "label", F.when(F.col("rating") >= 4, 1).when(F.col("rating") <= 2, 0))

    # Two random numbers per review, used for sampling (see common.py).
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
