import json
from textblob import TextBlob


def load_real_reviews(path: str) -> list[str]:
    """
    Load real-world reviews from a JSON file.

    The expected format is a JSON array of strings, where each string
    represents a single review text.

    These reviews are used ONLY for statistical comparison against
    synthetic data (no training or generation).
    """
    with open(path, "r") as f:
        return json.load(f)


def basic_stats(texts: list[str]) -> dict:
    """
    Compute basic statistical metrics for a collection of review texts.

    Metrics include:
    - Average review length (character count)
    - Average sentiment polarity
    - Ratio of positive reviews (polarity > 0.2)
    - Ratio of negative reviews (polarity < -0.2)

    These statistics provide a lightweight but effective way
    to compare real and synthetic datasets at an aggregate level.
    """
    lengths = [len(t) for t in texts]
    polarities = [TextBlob(t).sentiment.polarity for t in texts]

    return {
        "avg_length": sum(lengths) / len(lengths),
        "avg_sentiment": sum(polarities) / len(polarities),
        "positive_ratio": sum(1 for p in polarities if p > 0.2) / len(polarities),
        "negative_ratio": sum(1 for p in polarities if p < -0.2) / len(polarities),
    }


def compare_real_vs_synthetic(real_reviews: list[str], synthetic_reviews: list[dict]) -> dict:
    """
    Compare real-world reviews with synthetic reviews at a dataset level.

    Real reviews are passed directly as text strings.
    Synthetic reviews are extracted from structured records.

    The comparison focuses on high-level statistical alignment rather than
    content similarity, helping validate realism without data leakage.
    """
    real_stats = basic_stats(real_reviews)
    synthetic_stats = basic_stats([r["review"] for r in synthetic_reviews])

    return {
        "real": real_stats,
        "synthetic": synthetic_stats,
    }
