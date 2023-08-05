import os

from flask import Flask, render_template, request
from nltk.sentiment import SentimentIntensityAnalyzer

from sentiment_analyzer import analyze_sentiment_with_negative_words

sid = SentimentIntensityAnalyzer()

app = Flask(__name__)


@app.route('/')
def index():
    user = os.getlogin()  # Retrieve the User ID from the system
    return render_template('template.html', user=user)


@app.route('/analyze', methods=['POST'])
def analyze():
    user = request.form['user']
    message = request.form['message']

    # Perform sentiment analysis and get the sentiment score
    sentiment_score = perform_sentiment_analysis(message)

    return render_template('template.html', user=user, message=message, sentiment_score=sentiment_score)


def perform_sentiment_analysis(message):
    sentiment, sentiment_percentage, negative_words, ddd = analyze_sentiment_with_negative_words(message)

    sentiment_score = (str(sentiment) + " - " + str((round(sentiment_percentage))) + "%" + "\n\n" + str(ddd) +
                       "Negative Words:\n" + str(negative_words))

    return sentiment_score


if __name__ == '__main__':
    app.run(debug=True)
