# main.py

import nltk
from textblob import TextBlob
from collections import Counter
import re

# Download NLTK data if not already installed
nltk.download('punkt')

class TextAnalyzer:
    def __init__(self, text):
        self.text = text
        self.words = self._tokenize_words()
    
    def _tokenize_words(self):
        """Tokenizes the text into words and removes non-alphabetic characters."""
        words = nltk.word_tokenize(self.text)
        words = [word.lower() for word in words if word.isalpha()]
        return words
    
    def word_frequency(self):
        """Returns a dictionary of word frequencies in the text."""
        word_count = Counter(self.words)
        return word_count
    
    def sentiment_analysis(self):
        """Performs sentiment analysis on the text."""
        blob = TextBlob(self.text)
        sentiment = blob.sentiment
        return sentiment
    
    def summarize_text(self):
        """Generates a simple summary by extracting key sentences."""
        sentences = nltk.sent_tokenize(self.text)
        # Simple strategy: return the first two sentences
        summary = ' '.join(sentences[:2])  # Taking the first two sentences as summary
        return summary


def main():
    print("Welcome to the NLP Text Analyzer")
    print("Please enter a text for analysis:")
    text_input = input()

    analyzer = TextAnalyzer(text_input)
    
    # Word Frequency Analysis
    word_count = analyzer.word_frequency()
    print("\nWord Frequency Analysis:")
    for word, count in word_count.items():
        print(f"{word}: {count}")
    
    # Sentiment Analysis
    sentiment = analyzer.sentiment_analysis()
    print(f"\nSentiment Analysis:\nPolarity: {sentiment.polarity}, Subjectivity: {sentiment.subjectivity}")
    
    # Text Summarization
    summary = analyzer.summarize_text()
    print(f"\nText Summary: {summary}")

if __name__ == "__main__":
    main()
