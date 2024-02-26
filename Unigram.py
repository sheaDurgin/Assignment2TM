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
        self.get_term_freq_per_genre()
        self.term_prob_per_genre = {}
        self.get_term_prob_per_genre()

    def get_term_freq_per_genre(self):
        for genre, songs in self.genre_to_songs.items():
            for text in songs:  
                self.term_freq_per_genre[genre].update(preprocess(text))

    def get_term_prob_per_genre(self):
        for genre in self.genre_to_songs:
            total_terms = sum(self.term_freq_per_genre[genre].values())
            self.term_prob_per_genre[genre] = {term: freq / total_terms for term, freq in self.term_freq_per_genre[genre].items()}
    
    def calculate_prob(self, text):
        tokens = preprocess(text)
        results = {}
        for genre, term_prob in self.term_prob_per_genre.items():
            prob = 0.0
            for token in tokens:
                prob += math.log(term_prob.get(token, 1e-8))
            results[genre] = prob
        return results
    
    def predict(self, text):
        results = self.calculate_prob(text)
        return max(results, key=results.get)