"""Shared by the scripts: starting Spark, loading a sample, the TF-IDF steps."""

import json

import pandas as pd
from pyspark.ml import PipelineModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import IDF, ElementwiseProduct, HashingTF, NGram, Normalizer, Tokenizer
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

DATA = "data/reviews.parquet"
STATS = "results/prepare_stats.json"

# both engines hash words into 2^18 = 262,144 buckets
NUM_FEATURES = 2 ** 18


def get_spark(memory="8g"):
    """Start Spark on this machine with all CPU cores."""
    spark = (
        SparkSession.builder
        .master("local[*]")
        # on one machine everything runs in the driver, so this is all of Spark's memory
        .config("spark.driver.memory", memory)
        # the default of 200 is for clusters, on a laptop it makes hundreds of tiny tasks
        .config("spark.sql.shuffle.partitions", "16")
        # faster toPandas()
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")  # keep the console quiet
    return spark


def sample_fraction(size):
    """Number of reviews (or "all") -> fraction of the dataset."""
    if size == "all":
        return 1.0
    with open(STATS) as f:
        labelled_reviews = json.load(f)["labelled_rows"]
    return min(int(size) / labelled_reviews, 1.0)


# Sampling: prepare_data.py saved two random numbers per review (sample_key,
# split_key). "10% of the reviews" = sample_key < 0.1, and the test set is
# always split_key < 0.2. Same numbers for pandas and Spark, so the same rows.

def load_sample_pandas(size, columns=("clean_text", "label")):
    """Sample of the labelled reviews as pandas. Returns (train, test)."""
    df = pd.read_parquet(
        DATA,
        columns=list(columns) + ["split_key"],
        filters=[("sample_key", "<", sample_fraction(size)), ("label", "in", [0, 1])],
    )
    train = df[df["split_key"] >= 0.2]
    test = df[df["split_key"] < 0.2]
    return train, test


def load_sample_spark(spark, size):
    """Same sample and split as load_sample_pandas, as Spark DataFrames."""
    df = spark.read.parquet(DATA)
    df = df.filter(F.col("sample_key") < sample_fraction(size))
    df = df.filter(F.col("label").isNotNull())
    df = df.select("clean_text", "label", "split_key")
    train = df.filter(F.col("split_key") >= 0.2)
    test = df.filter(F.col("split_key") < 0.2)
    return train, test


def add_terms(df):
    """Words plus pairs of words, so "not good" is its own term.

    "not good game" -> ["not", "good", "game", "not good", "good game"]
    """
    df = Tokenizer(inputCol="clean_text", outputCol="words").transform(df)
    df = NGram(n=2, inputCol="words", outputCol="pairs").transform(df)
    return df.withColumn("terms", F.concat("words", "pairs"))


def fit_tfidf(train):
    """Learn the IDF of every bucket on the training reviews.

    transform() on the result adds a "features" column, computed like
    scikit-learn's HashingVectorizer + TfidfTransformer.
    """
    hashing = HashingTF(inputCol="terms", outputCol="counts", numFeatures=NUM_FEATURES)
    idf = IDF(inputCol="counts").fit(hashing.transform(train))
    # Spark's IDF is log((n + 1) / (df + 1)), scikit-learn's is that plus 1.
    # Adding 1 here makes the two give the same numbers.
    idf_plus_one = Vectors.dense(idf.idf.toArray() + 1)
    # PipelineModel = steps that are already trained, run one after the other
    return PipelineModel([
        hashing,
        ElementwiseProduct(inputCol="counts", outputCol="tfidf", scalingVec=idf_plus_one),
        # length 1 per review, so long reviews don't count more
        Normalizer(inputCol="tfidf", outputCol="features"),
    ])


def spark_logistic_regression(train_rows):
    """Spark's logistic regression, set up to match scikit-learn's
    LogisticRegression(max_iter=1000, class_weight="balanced").

    Two of Spark's defaults gave a much lower score on the same reviews:
    - regParam is 0. scikit-learn penalizes large weights (C=1), and Spark
      averages the loss while scikit-learn sums it, so the same penalty is
      1 / number of training reviews.
    - standardization rescales features before the penalty, so rare words are
      barely penalized and the model overfits.
    """
    return LogisticRegression(weightCol="weight", regParam=1.0 / train_rows,
                              standardization=False)


def add_class_weights(train):
    """Add a weight column so both classes count equally (the formula of
    scikit-learn's class_weight="balanced"). Returns the DataFrame and its row count."""
    positives = train.filter(F.col("label") == 1).count()
    negatives = train.filter(F.col("label") == 0).count()
    total = positives + negatives
    weight = F.when(F.col("label") == 1, total / (2 * positives)).otherwise(total / (2 * negatives))
    return train.withColumn("weight", weight), total
