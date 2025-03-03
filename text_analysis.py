import nltk
from textblob import TextBlob
from collections import Counter

nltk.download('punkt')

def analyze_sentiment(text):
    """Perform sentiment analysis on the text."""
    blob = TextBlob(text)
    return blob.sentiment.polarity  # Returns score between -1 and 1

def word_frequency(text, top_n=10):
    """Calculate word frequency."""
    words = nltk.word_tokenize(text.lower())
    word_counts = Counter(words)
    return word_counts.most_common(top_n)
