"""Where the model goes wrong, and how much keeping "not" helps.

    python error_analysis.py                 (sample of 200,000 reviews)
    python error_analysis.py --size 50000

Uses a real vocabulary (TfidfVectorizer) instead of hashing, so the weights
can be read as words. Writes results/error_analysis.md.
"""

import argparse

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline

from cleaning import contains_negation, remove_negations
from common import load_sample_pandas


def build_model():
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=5),  # ignore terms seen in fewer than 5 reviews
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )


def shorten(text, length=200):
    """Cut a raw review down for the report."""
    text = " ".join(text.replace("<br />", " ").split())
    return text if len(text) <= length else text[:length] + "..."


def list_reviews(reviews):
    """One markdown bullet per review: stars, predicted probability, text."""
    lines = []
    for _, row in reviews.iterrows():
        lines.append(f'- {int(row["rating"])} stars, P(positive) = {row["prob_positive"]:.2f}: '
                     f'"{shorten(row["text"])}"')
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", default="200000", help='number of reviews, or "all"')
    args = parser.parse_args()

    train, test = load_sample_pandas(args.size, columns=["text", "clean_text", "rating", "label"])
    test = test.copy()

    model = build_model()
    model.fit(train["clean_text"], train["label"])
    test["predicted"] = model.predict(test["clean_text"])
    test["prob_positive"] = model.predict_proba(test["clean_text"])[:, 1]
    test["wrong"] = test["predicted"] != test["label"]
    macro_f1 = f1_score(test["label"], test["predicted"], average="macro")

    report = [f"Sample: {len(train):,} training and {len(test):,} test reviews. "
              f"Macro F1: {macro_f1:.3f}"]

    # 1. Which star ratings does the model get wrong most often?
    by_stars = test.groupby("rating").agg(reviews=("wrong", "size"), error_rate=("wrong", "mean"))
    by_stars["error_rate"] = (100 * by_stars["error_rate"]).round(1).astype(str) + "%"
    report += ["### Error rate by star rating", by_stars.to_markdown()]

    # 2. Which words and pairs pushed the model most towards each answer?
    vectorizer = model.named_steps["tfidfvectorizer"]
    classifier = model.named_steps["logisticregression"]
    weights = pd.Series(classifier.coef_[0], index=vectorizer.get_feature_names_out())
    strongest = pd.DataFrame({
        "most negative": weights.nsmallest(15).index,
        "most positive": weights.nlargest(15).index,
    })
    report += ["### Strongest words and pairs", strongest.to_markdown(index=False)]

    # 3. the mistakes it was surest about, often text and stars that don't match
    false_positives = test[(test["label"] == 0) & test["wrong"]].nlargest(5, "prob_positive")
    false_negatives = test[(test["label"] == 1) & test["wrong"]].nsmallest(5, "prob_positive")
    report += ["### Most confident mistakes",
               "Negative reviews (1-2 stars) the model was surest were positive:",
               list_reviews(false_positives),
               "Positive reviews (4-5 stars) the model was surest were negative:",
               list_reviews(false_negatives)]

    # 4. same model again without "no", "nor" and "not"
    model_without = build_model()
    model_without.fit(train["clean_text"].map(remove_negations), train["label"])
    test["predicted_without"] = model_without.predict(test["clean_text"].map(remove_negations))
    macro_f1_without = f1_score(test["label"], test["predicted_without"], average="macro")

    with_negation = test[test["clean_text"].map(contains_negation)]
    accuracy_kept = (with_negation["predicted"] == with_negation["label"]).mean()
    accuracy_removed = (with_negation["predicted_without"] == with_negation["label"]).mean()

    comparison = pd.DataFrame({
        "negation words": ["kept", "removed"],
        "macro F1 (all test reviews)": [round(macro_f1, 3), round(macro_f1_without, 3)],
        "accuracy on reviews with a negation": [round(accuracy_kept, 3), round(accuracy_removed, 3)],
    })
    share = len(with_negation) / len(test)
    report += ["### Keeping vs removing negation words", comparison.to_markdown(index=False),
               f"{share:.1%} of the test reviews contain no, nor or not."]

    with open("results/error_analysis.md", "w") as f:
        f.write("\n\n".join(report) + "\n")
    print("\n\n".join(report))


if __name__ == "__main__":
    main()
