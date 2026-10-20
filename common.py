"""Code shared by the scripts: starting Spark, loading a sample of the data,
and the steps that turn text into numbers in Spark."""

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

# Both engines hash words into the same number of buckets: 2^18 = 262,144.
NUM_FEATURES = 2 ** 18


def get_spark(memory="8g"):
    """Start Spark on this machine, using all the CPU cores."""
    spark = (
        SparkSession.builder
        .master("local[*]")
        # On one machine Spark does all its work in this "driver" process,
        # so this is how much memory Spark gets.
        .config("spark.driver.memory", memory)
        # The default of 200 is meant for clusters. On a laptop it creates
        # hundreds of tiny tasks.
        .config("spark.sql.shuffle.partitions", "16")
        # Makes converting Spark results to pandas faster.
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")  # hide Spark's info messages
    return spark


def sample_fraction(size):
    """Turn a number of reviews (or "all") into a fraction of the dataset."""
    if size == "all":
        return 1.0
    with open(STATS) as f:
        labelled_reviews = json.load(f)["labelled_rows"]
    return min(int(size) / labelled_reviews, 1.0)


# How the sampling works: prepare_data.py gave every review two random numbers
# between 0 and 1, sample_key and split_key, and saved them with the data.
# "Take 10% of the reviews" means "keep the rows where sample_key < 0.1".
# Because the numbers are saved, pandas and Spark get exactly the same rows,
# and split_key < 0.2 always puts the same 20% in the test set.

def load_sample_pandas(size, columns=("clean_text", "label")):
    """Load a sample of the labelled reviews with pandas. Returns (train, test)."""
    df = pd.read_parquet(
        DATA,
        columns=list(columns) + ["split_key"],
        filters=[("sample_key", "<", sample_fraction(size)), ("label", "in", [0, 1])],
    )
    train = df[df["split_key"] >= 0.2]
    test = df[df["split_key"] < 0.2]
    return train, test


def load_sample_spark(spark, size):
    """The same sample and split as load_sample_pandas, as Spark DataFrames."""
    df = spark.read.parquet(DATA)
    df = df.filter(F.col("sample_key") < sample_fraction(size))
    df = df.filter(F.col("label").isNotNull())
    df = df.select("clean_text", "label", "split_key")
    train = df.filter(F.col("split_key") >= 0.2)
    test = df.filter(F.col("split_key") < 0.2)
    return train, test


def add_terms(df):
    """Split each review into words and pairs of words.

    "not good game" -> ["not", "good", "game", "not good", "good game"]
    The pairs let the model see "not good" as its own term.
    """
    df = Tokenizer(inputCol="clean_text", outputCol="words").transform(df)
    df = NGram(n=2, inputCol="words", outputCol="pairs").transform(df)
    return df.withColumn("terms", F.concat("words", "pairs"))


def fit_tfidf(train):
    """Learn the IDF of every bucket from the training reviews.

    Returns a model whose transform() adds a "features" column, computed the
    same way as scikit-learn's HashingVectorizer + TfidfTransformer.
    """
    hashing = HashingTF(inputCol="terms", outputCol="counts", numFeatures=NUM_FEATURES)
    idf = IDF(inputCol="counts").fit(hashing.transform(train))
    # Spark's IDF is log((n + 1) / (df + 1)). scikit-learn's is the same plus 1,
    # so a word that is in almost every review still counts a little. Adding
    # the 1 here makes both engines compute the same numbers.
    idf_plus_one = Vectors.dense(idf.idf.toArray() + 1)
    # A PipelineModel runs the steps one after the other, like a Pipeline that
    # has already been trained.
    return PipelineModel([
        hashing,
        ElementwiseProduct(inputCol="counts", outputCol="tfidf", scalingVec=idf_plus_one),
        # scale each review's vector to length 1, so long reviews don't count more
        Normalizer(inputCol="tfidf", outputCol="features"),
    ])


def spark_logistic_regression(train_rows):
    """Spark's logistic regression, set up to train the same model as
    scikit-learn's LogisticRegression(max_iter=1000, class_weight="balanced").

    Two of Spark's defaults are different, and with them Spark scored much
    lower than scikit-learn on the same reviews:
    - regParam: scikit-learn adds a penalty for large weights (C=1), which
      stops the model from memorising rare words. Spark adds none. Spark
      averages the loss over the reviews while scikit-learn adds it up, so
      the same penalty in Spark is 1 / number of training reviews.
    - standardization: Spark rescales every feature before applying the
      penalty. Rare words then get almost no penalty, and the model overfits.
    """
    return LogisticRegression(weightCol="weight", regParam=1.0 / train_rows,
                              standardization=False)


def add_class_weights(train):
    """Add a weight column so positive and negative reviews count equally.

    Most reviews are positive, so without weights a model can score well by
    mostly answering "positive". Same formula as scikit-learn's
    class_weight="balanced". Returns the DataFrame and its number of rows.
    """
    positives = train.filter(F.col("label") == 1).count()
    negatives = train.filter(F.col("label") == 0).count()
    total = positives + negatives
    weight = F.when(F.col("label") == 1, total / (2 * positives)).otherwise(total / (2 * negatives))
    return train.withColumn("weight", weight), total
