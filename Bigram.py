import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
l = WordNetLemmatizer()
nltk.download('punkt')
nltk.download('stopwords')
stopwords = set(stopwords.words('english'))
import string
import math
import re
from collections import Counter

def preprocess(text):
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text.strip())
    text = ''.join([c.lower() for c in text if c not in string.punctuation])
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stopwords]
    tokens = [l.lemmatize(token) for token in tokens]
    return tokens

class Bigram:
    def __init__(self, genre_to_songs):
        self.genre_to_songs = genre_to_songs
        self.term_freq_per_genre = {genre: Counter() for genre in genre_to_songs.keys()}
        self.total_terms = {}
        self.term_prob_per_genre = {}
        self.get_term_freq_per_genre()
        self.get_term_prob_per_genre()

    # Get freq for tokens in each genre
    def get_term_freq_per_genre(self):
        for genre, songs in self.genre_to_songs.items():
            for text in songs:     
                tokens = preprocess(text)
                for prev, token in zip(tokens[:-1], tokens[1:]):
                    self.term_freq_per_genre[genre][(prev, token)] += 1

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
            for prev, token in zip(tokens[:-1], tokens[1:]):
                prob += math.log(term_prob.get((prev, token), 1e-7))
            results[genre] = prob
        return results
    
    # Predict a genre
    def predict(self, text):
        results = self.calculate_prob(text)
        return max(results, key=results.get)