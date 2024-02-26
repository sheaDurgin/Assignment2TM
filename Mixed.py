from Unigram import Unigram
from Bigram import Bigram

class Mixed:
    def __init__(self, genre_to_songs, split):
        self.u_model = Unigram(genre_to_songs)
        self.b_model = Bigram(genre_to_songs)
        self.split = split

    # Calculate probability for each genre given a text  
    def calculate_prob(self, input_text):
        unigram_results = self.u_model.calculate_prob(input_text)
        bigram_results = self.b_model.calculate_prob(input_text)

        mixed_results = {}
        for genre in self.u_model.genre_to_songs:
            mixed_results[genre] = (self.split * unigram_results[genre]) + ((1 - self.split) * bigram_results[genre])

        return mixed_results
    
    # Predict a genre
    def predict(self, input_text):
        results = self.calculate_prob(input_text)
        return max(results, key=results.get)
    
    # For all models, calculate probability for each genre given a text  
    def calculate_prob_and_return_all(self, input_text):
        unigram_results = self.u_model.calculate_prob(input_text)
        bigram_results = self.b_model.calculate_prob(input_text)

        mixed_results = {}
        for genre in self.u_model.genre_to_songs:
            mixed_results[genre] = (self.split * unigram_results[genre]) + ((1 - self.split) * bigram_results[genre])

        return [mixed_results, unigram_results, bigram_results]
    
    # For all models, predict a genre
    def predict_and_return_all(self, input_text):
        all_results = self.calculate_prob_and_return_all(input_text)
        return [max(results, key=results.get) for results in all_results]