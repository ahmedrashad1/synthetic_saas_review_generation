from textblob import TextBlob

def rating_sentiment_ok(text: str, rating: int,
                        low_rating_positive_cutoff: float,
                        high_rating_negative_cutoff: float) -> bool:
    """
    Validate alignment between a review's star rating and its textual sentiment.

    This function prevents unrealistic cases such as:
    - Very positive language paired with low ratings (1–2 stars)
    - Strongly negative language paired with high ratings (4–5 stars)

    Sentiment polarity is computed using TextBlob and compared against
    configurable thresholds to allow controlled flexibility.

    Parameters:
    - text: The review text to analyze
    - rating: Star rating (1–5)
    - low_rating_positive_cutoff: Maximum allowed polarity for low ratings
    - high_rating_negative_cutoff: Minimum allowed polarity for high ratings

    Returns:
    - True if sentiment and rating are logically consistent
    - False if the combination is deemed unrealistic
    """
    polarity = TextBlob(text).sentiment.polarity

    if rating <= 2 and polarity > low_rating_positive_cutoff:
        return False

    if rating >= 4 and polarity < high_rating_negative_cutoff:
        return False

    return True
