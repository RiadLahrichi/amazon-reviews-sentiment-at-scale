"""A few questions about the reviews, answered with Spark SQL on the full data.

    python insights.py

Writes results/insights.md.
"""

from common import DATA, get_spark

QUESTIONS = {
    "Review length and helpful votes by star rating": """
        SELECT CAST(rating AS INT) AS stars,
               COUNT(*) AS reviews,
               ROUND(AVG(word_count), 1) AS avg_words,
               ROUND(AVG(helpful_vote), 2) AS avg_helpful_votes
        FROM reviews
        GROUP BY rating
        ORDER BY stars
    """,

    "Verified vs unverified purchases": """
        SELECT verified_purchase,
               COUNT(*) AS reviews,
               ROUND(AVG(rating), 2) AS avg_rating,
               ROUND(100 * AVG(CASE WHEN rating <= 2 THEN 1 ELSE 0 END), 1) AS pct_negative
        FROM reviews
        GROUP BY verified_purchase
    """,

    # polarising = lots of 1-star and lots of 5-star reviews, so a high standard deviation
    "Most polarising products (at least 100 reviews)": """
        SELECT parent_asin,
               COUNT(*) AS reviews,
               ROUND(AVG(rating), 2) AS avg_rating,
               ROUND(STDDEV(rating), 2) AS rating_std,
               ROUND(100 * AVG(CASE WHEN rating = 1 THEN 1 ELSE 0 END), 1) AS pct_1_star,
               ROUND(100 * AVG(CASE WHEN rating = 5 THEN 1 ELSE 0 END), 1) AS pct_5_star
        FROM reviews
        GROUP BY parent_asin
        HAVING COUNT(*) >= 100
        ORDER BY rating_std DESC
        LIMIT 10
    """,

    # timestamp is in milliseconds
    "Reviews and average rating by year": """
        SELECT YEAR(FROM_UNIXTIME(timestamp / 1000)) AS year,
               COUNT(*) AS reviews,
               ROUND(AVG(rating), 2) AS avg_rating
        FROM reviews
        GROUP BY year
        ORDER BY year
    """,
}


def main():
    spark = get_spark()
    # make the data available as a table called "reviews"
    spark.read.parquet(DATA).createOrReplaceTempView("reviews")

    sections = []
    for title, query in QUESTIONS.items():
        print(f"running: {title}")
        answer = spark.sql(query).toPandas()
        sections.append(f"### {title}\n\n{answer.to_markdown(index=False)}\n")

    with open("results/insights.md", "w") as f:
        f.write("\n".join(sections))


if __name__ == "__main__":
    main()
