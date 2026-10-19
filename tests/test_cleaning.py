import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from cleaning import NEGATIONS, STOP_WORDS, clean_text, contains_negation, remove_negations


@pytest.fixture(scope="module")
def spark():
    spark = SparkSession.builder.master("local[1]").getOrCreate()
    yield spark
    spark.stop()


def run_clean_text(spark, text):
    df = spark.createDataFrame([(text,)], "text string")
    return df.select(clean_text(F.col("text"))).first()[0]


@pytest.mark.parametrize("raw, expected", [
    ("Great<br />product", "great product"),   # tags must not glue words together
    ("I don't like it", "i do not like it"),
    ("It doesn’t work", "it does not work"),   # curly apostrophe
    ("I won't buy again", "i will not buy again"),
    ("Can't recommend", "can not recommend"),
    ("see https://amzn.to/x for 2 pics!!!", "see for pics"),
])
def test_clean_text(spark, raw, expected):
    assert run_clean_text(spark, raw) == expected


def test_negations_are_not_stop_words():
    # the bug in the first version of this project
    assert not NEGATIONS & set(STOP_WORDS)


def test_remove_negations():
    assert remove_negations("not good no fun nothing") == "good fun nothing"
    assert contains_negation("not good")
    assert not contains_negation("nothing good")
