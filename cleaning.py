"""Text cleaning rules, used by prepare_data.py.

The cleaning uses Spark's built-in functions instead of a normal Python
function applied to each review. Built-in functions run inside Spark itself,
which is much faster than sending millions of reviews to Python and back.
"""

from pyspark.sql import functions as F

# Words that flip the meaning of what follows. The first version of this
# project removed them as stop words, so "not good" turned into "good".
NEGATIONS = {"no", "nor", "not"}

# The same stop word list as the first version, minus the negation words.
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
    # Reviews use <br /> between sentences. Replacing tags with a space
    # (not with nothing) stops "great<br />product" becoming "greatproduct".
    text = F.regexp_replace(text, r"<[^>]+>", " ")
    text = F.regexp_replace(text, r"https?://\S+|www\.\S+", " ")  # links
    # Expand negative contractions before removing punctuation, so the
    # "not" survives. Reviews use both ' and ’ as apostrophes.
    text = F.regexp_replace(text, r"won[’']t", "will not")
    text = F.regexp_replace(text, r"can[’']t", "can not")
    text = F.regexp_replace(text, r"n[’']t\b", " not")  # don't, isn't, didn't...
    text = F.regexp_replace(text, r"[^a-z\s]", " ")  # remove everything except letters
    text = F.regexp_replace(text, r"\s+", " ")  # many spaces -> one space
    return F.trim(text)


def remove_negations(text):
    """Drop "no", "nor" and "not" from a cleaned review.

    Only used by error_analysis.py, to measure how much these words help.
    """
    return " ".join(word for word in text.split() if word not in NEGATIONS)


def contains_negation(text):
    return any(word in NEGATIONS for word in text.split())
