import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

analyzer = SentimentIntensityAnalyzer()


def get_sentiment_features(text: str) -> np.ndarray:
    """
    Returns a 6-dim sentiment feature vector:
    [vader_pos, vader_neg, vader_neu, vader_compound, tb_polarity, tb_subjectivity]
    """
    text = str(text)
    vader = analyzer.polarity_scores(text)
    blob = TextBlob(text)

    features = np.array([
        vader["pos"],
        vader["neg"],
        vader["neu"],
        vader["compound"],
        blob.sentiment.polarity,
        blob.sentiment.subjectivity,
    ], dtype=np.float32)

    return features


def build_sentiment_features(texts):
    return np.stack([get_sentiment_features(t) for t in texts])