from collections import Counter
from textblob import TextBlob


def analyze_sentiment(reviews: list[dict]) -> dict:
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
    counter = Counter(r["rating"] for r in reviews)
    total = len(reviews)
    return {str(k): v / total for k, v in counter.items()}


def analyze_personas(reviews: list[dict]) -> dict:
    counter = Counter(r["persona"] for r in reviews)
    total = len(reviews)
    return {k: v / total for k, v in counter.items()}
