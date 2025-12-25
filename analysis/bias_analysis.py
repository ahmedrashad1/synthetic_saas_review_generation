from collections import Counter
from textblob import TextBlob


def analyze_sentiment(reviews: list[dict]) -> dict:
    """
    Analyze the sentiment distribution of a list of reviews.

    Each review's text is analyzed using TextBlob polarity:
    - polarity > 0.2  → positive
    - polarity < -0.2 → negative
    - otherwise       → neutral

    Returns normalized ratios (percentages) rather than raw counts
    to make results independent of dataset size.
    """
    sentiments = {"positive": 0, "neutral": 0, "negative": 0}

    for r in reviews:
        polarity = TextBlob(r["review"]).sentiment.polarity
        if polarity > 0.2:
            sentiments["positive"] += 1
        elif polarity < -0.2:
            sentiments["negative"] += 1
        else:
            sentiments["neutral"] += 1

    total = len(reviews)
    return {k: v / total for k, v in sentiments.items()}


def analyze_ratings(reviews: list[dict]) -> dict:
    """
    Analyze the distribution of star ratings (1–5).

    Counts how often each rating appears and converts
    the counts into normalized proportions.

    Output keys are strings for JSON compatibility.
    """
    counter = Counter(r["rating"] for r in reviews)
    total = len(reviews)
    return {str(k): v / total for k, v in counter.items()}


def analyze_personas(reviews: list[dict]) -> dict:
    """
    Analyze how reviews are distributed across personas.

    Ensures the synthetic dataset does not over-represent
    a single user type (e.g. developers only).

    Returns normalized proportions per persona.
    """
    counter = Counter(r["persona"] for r in reviews)
    total = len(reviews)
    return {k: v / total for k, v in counter.items()}
