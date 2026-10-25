"""Text cleaning rules, used by prepare_data.py.

Built from Spark's own functions, because a Python function per review would
mean sending millions of rows to Python and back.
"""

from pyspark.sql import functions as F

# words that flip the meaning. Version 1 removed them as stop words, so "not good" became "good"
NEGATIONS = {"no", "nor", "not"}

# the version 1 stop word list, without the negation words
STOP_WORDS = [
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
    "she", "her", "hers", "herself", "it", "its", "itself", "they", "them",
    "their", "theirs", "themselves", "what", "which", "who", "whom", "this",
    "that", "these", "those", "am", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "having", "do", "does", "did", "doing",
    "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between",
    "through", "during", "before", "after", "above", "below", "to", "from",
    "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "both", "each", "few", "more", "most", "other", "some",
    "such", "only", "own", "same", "so", "than", "too", "very", "s", "t",
    "can", "will", "just", "don", "should", "now", "d", "ll", "m", "o", "re",
    "ve", "y", "ain", "aren", "couldn", "didn", "doesn", "hadn", "hasn",
    "haven", "isn", "ma", "mightn", "mustn", "needn", "shan", "shouldn",
    "wasn", "weren", "won", "wouldn",
]


def clean_text(text):
    """Lowercase a review and keep only words.

    "Don't buy it!<br />Broke in 2 days" becomes "do not buy it broke in days".
    """
    text = F.lower(text)
    # replace tags with a space, not nothing, or "great<br />product" becomes "greatproduct"
    text = F.regexp_replace(text, r"<[^>]+>", " ")
    text = F.regexp_replace(text, r"https?://\S+|www\.\S+", " ")  # links
    # expand contractions before punctuation goes, so the "not" survives (reviews use ' and ’)
    text = F.regexp_replace(text, r"won[’']t", "will not")
    text = F.regexp_replace(text, r"can[’']t", "can not")
    text = F.regexp_replace(text, r"n[’']t\b", " not")  # don't, isn't, didn't...
    text = F.regexp_replace(text, r"[^a-z\s]", " ")  # letters only
    text = F.regexp_replace(text, r"\s+", " ")  # one space
    return F.trim(text)


def remove_negations(text):
    """Drop "no", "nor" and "not". Only used by error_analysis.py to see how much they help."""
    return " ".join(word for word in text.split() if word not in NEGATIONS)


def contains_negation(text):
    return any(word in NEGATIONS for word in text.split())
