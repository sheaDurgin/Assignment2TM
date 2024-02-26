import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
p = PorterStemmer()
nltk.download('punkt')
nltk.download('stopwords')
stopwords = set(stopwords.words('english'))
import string
import math
import re
from collections import Counter

def preprocess(text):
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text.strip())
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in stopwords and token not in string.punctuation]
    tokens = [p.stem(token) for token in tokens]
    return tokens

class Unigram:
    def __init__(self, genre_to_songs):
        self.genre_to_songs = genre_to_songs
        self.term_freq_per_genre = {genre: Counter() for genre in self.genre_to_songs}
        self.total_terms = {}
        self.term_prob_per_genre = {}
        self.get_term_freq_per_genre()
        self.get_term_prob_per_genre()

    # Get freq for tokens in each genre
    def get_term_freq_per_genre(self):
        for genre, songs in self.genre_to_songs.items():
            for text in songs:
                self.term_freq_per_genre[genre].update(preprocess(text))

    # Convert freq to prob for each genres tokens
    def get_term_prob_per_genre(self):
        for genre in self.genre_to_songs:
            self.total_terms[genre] = sum(self.term_freq_per_genre[genre].values()) + len(self.term_freq_per_genre[genre])
            self.term_prob_per_genre[genre] = {term: (freq + 1) / self.total_terms[genre] for term, freq in self.term_freq_per_genre[genre].items()}
    
    # Calculate probability for each genre given a text
    def calculate_prob(self, text):
        tokens = preprocess(text)
        results = {}
        for genre, term_prob in self.term_prob_per_genre.items():
            prob = 0.0
            for token in tokens:
                prob += math.log(term_prob.get(token, 1e-8))
            results[genre] = prob
        return results
    
    # Predict a genre
    def predict(self, text):
        results = self.calculate_prob(text)
        return max(results, key=results.get)