from Mixed import Mixed
import os
import csv
import sys
import random
from sklearn.metrics import f1_score
from tabulate import tabulate
from scipy import stats
from itertools import combinations

average = 'weighted'
model_names = ['Mixed', 'Unigram', 'Bigram']

# Returns a training and validation set
def split_files_random(genre_to_songs, train_ratio=0.9, random_seed=None):
    if random_seed is not None:
        random.seed(random_seed)

    train_genre_to_songs = {}
    dev_genre_to_songs = {}
    
    for genre, songs in genre_to_songs.items():
        random.shuffle(songs)
        train_split = int(len(songs) * train_ratio)
        train_genre_to_songs[genre] = songs[:train_split]
        dev_genre_to_songs[genre] = songs[train_split:]

    return train_genre_to_songs, dev_genre_to_songs

def color_cell(gold, prediction):
    if gold == prediction:
        return "\033[92m{}\033[0m".format(prediction)  # Green color
    else:
        return "\033[91m{}\033[0m".format(prediction)  # Red color
    
def print_tables(scores):
    f1_table_data = []
    for name in model_names:
        f1 = f1_score(scores[name]['y_true'], scores[name]['y_pred'], average=average)
        f1_table_data.append([name, f1])
    
    headers = ["Model", "F1-Score"]
    print(tabulate(f1_table_data, headers=headers, tablefmt="pretty"))

    table_data = []
    total_correct_row = ["Total", 0, 0, 0]
    for gold, *predictions in zip(scores['Mixed']['y_true'], *[scores[name]['y_pred'] for name in model_names]):
        colored_predictions = [color_cell(gold, pred) for pred in predictions]
        row = [gold] + colored_predictions
        table_data.append(row)

        for i, pred in enumerate(predictions, start=1):
            total_correct_row[i] += 1 if gold == pred else 0

    table_data.append(total_correct_row)

    headers = ['Gold'] + [name for name in model_names]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

def mcnemar_test(table):
    b = table[0][1]  # Instances misclassified by model 2 but not by model 1
    c = table[1][0]  # Instances misclassified by model 1 but not by model 2

    chi2 = ((abs(b - c) - 1) ** 2) / (b + c)
    # Compute p-value using chi-square distribution with 1 degree of freedom
    p_value = stats.chi2.sf(chi2, 1)
    
    return chi2, p_value

def pairwise_mcnemar_test(scores):
    for model1, model2 in combinations(model_names, 2):
        contingency_table = [[0, 0], [0, 0]]  # Initialize 2x2 contingency table

        for pred1_correct, pred2_correct in zip(scores[model1]['correct'], scores[model2]['correct']):
            contingency_table[pred1_correct ^ 1][pred2_correct ^ 1] += 1

        chi2, p_value = mcnemar_test(contingency_table)
        
        print(f"Comparing {model1} and {model2}:")
        print("  - Chi-Square Statistic:", chi2)
        print("  - p-value:", p_value)
        if p_value < 0.05:
            print("  - Significant difference (p < 0.05)\n")
        else:
            print("  - No significant difference (p >= 0.05)\n")

# Get f1 scores for each model on the test set
def test(model):
    with open('test.tsv', 'r', newline='') as tsvfile:
        tsv_reader = csv.reader(tsvfile, delimiter='\t')
        next(tsv_reader)
        lines = [line for line in tsv_reader]

    scores = {score_type: {'y_true': [], 'y_pred': [], 'correct': []} for score_type in model_names}

    for line in lines:
        _, text, gold_genre = line

        for name, pred_genre in zip(model_names, model.predict_and_return_all(text)):
            scores[name]['y_true'].append(gold_genre)
            scores[name]['y_pred'].append(pred_genre)
            scores[name]['correct'].append(1 if gold_genre == pred_genre else 0)

    print_tables(scores)
    pairwise_mcnemar_test(scores)

# Get song data for each genre
def get_data(dir_path):
    genre_to_songs = {}
    for genre in os.listdir(dir_path):
        genre_dir_path = os.path.join(dir_path, genre)

        files = os.listdir(genre_dir_path)
        song_paths = [os.path.join(genre_dir_path, file) for file in files]
        genre_to_songs[genre] = []
        for song in song_paths:
            with open(song, 'r') as f:
                text = f.read()
            genre_to_songs[genre].append(text.strip())
    
    return genre_to_songs

# Find optimal lambda value using the training and validation sets
def get_best_model_split(genre_to_songs):
    train_genre_to_songs, dev_genre_to_songs = split_files_random(genre_to_songs, random_seed=42)

    best_split = None
    best_f1 = -1
    for split in [i * 0.1 for i in range(1, 10)]:
        mixed_model = Mixed(train_genre_to_songs, split)
        scores = {'y_true': [], 'y_pred': []}

        for genre, songs in dev_genre_to_songs.items():
            for song in songs:
                pred_genre = mixed_model.predict(song)
                scores['y_true'].append(genre)
                scores['y_pred'].append(pred_genre)

        f1 = f1_score(scores['y_true'], scores['y_pred'], average=average)

        if f1 > best_f1:
            best_f1 = f1
            best_split = split
    
    return best_split

def main():
    dir_path = 'TM_CA1_Lyrics'
    genre_to_songs = get_data(dir_path)

    args = sys.argv[1:]
    if '--split' in args:
        split_index = args.index('--split')
        split = args[split_index + 1] if split_index + 1 < len(args) else None

    if '--split' not in args or split is None:
        split = get_best_model_split(genre_to_songs)

    split = float(split)
    print(f"We are using a Unigram Bigram spit of ({split}, {1-split})")

    mixed = Mixed(genre_to_songs, split)
    test(mixed)

if __name__ == '__main__':
    main()