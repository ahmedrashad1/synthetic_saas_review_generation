import json
from textblob import TextBlob


def load_real_reviews(path: str) -> list[str]:
    with open(path, "r") as f:
        return json.load(f)


def basic_stats(texts: list[str]) -> dict:
    lengths = [len(t) for t in texts]
    polarities = [TextBlob(t).sentiment.polarity for t in texts]

    return {
        "avg_length": sum(lengths) / len(lengths),
        "avg_sentiment": sum(polarities) / len(polarities),
        "positive_ratio": sum(1 for p in polarities if p > 0.2) / len(polarities),
        "negative_ratio": sum(1 for p in polarities if p < -0.2) / len(polarities),
    }


def compare_real_vs_synthetic(real_reviews: list[str], synthetic_reviews: list[dict]) -> dict:
    real_stats = basic_stats(real_reviews)
    synthetic_stats = basic_stats([r["review"] for r in synthetic_reviews])

    return {
        "real": real_stats,
        "synthetic": synthetic_stats,
    }
