from textblob import TextBlob

def rating_sentiment_ok(text: str, rating: int,
                        low_rating_positive_cutoff: float,
                        high_rating_negative_cutoff: float) -> bool:
    polarity = TextBlob(text).sentiment.polarity

    if rating <= 2 and polarity > low_rating_positive_cutoff:
        return False

    if rating >= 4 and polarity < high_rating_negative_cutoff:
        return False

    return True
