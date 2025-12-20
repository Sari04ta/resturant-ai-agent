from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def compute_sentiment(text: str) -> float:
    """
    Returns compound sentiment score in range [-1, 1]
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0

    return analyzer.polarity_scores(text)["compound"]
