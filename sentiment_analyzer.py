import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize

# Download required resources for sentiment analysis
nltk.download('vader_lexicon')
nltk.download('punkt')


def analyze_sentiment_with_negative_words(text):
    # Initialize the SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()

    # Get the sentiment score for the text
    sentiment_score = sid.polarity_scores(text)
    ddd = sentiment_score['compound']

    # Determine the overall sentiment category
    if sentiment_score['compound'] >= 0.05:
        sentiment = 'Positive'
    elif sentiment_score['compound'] <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'

    # Calculate the percentage distribution of each sentiment category
    total_score = abs(sentiment_score['pos']) + abs(sentiment_score['neu']) + abs(sentiment_score['neg'])
    sentiment_sco = (sentiment_score["compound"] / total_score) * 100

    # Get all the negative words in the text
    words = word_tokenize(text)
    negative_words = [word.lower() for word in words if sid.polarity_scores(word)['compound'] <= -0.05]

    return sentiment, sentiment_sco, negative_words, ddd
