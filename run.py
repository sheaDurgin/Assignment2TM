from Mixed import Mixed
import os
import csv
import sys
import random
from sklearn.metrics import f1_score

average = 'weighted'

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

# Get f1 scores for each model on the test set
def test(model):
    with open('test.tsv', 'r', newline='') as tsvfile:
        tsv_reader = csv.reader(tsvfile, delimiter='\t')
        next(tsv_reader)
        lines = [line for line in tsv_reader]

    model_names = ['Mixed', 'Unigram', 'Bigram']
    scores = {score_type: {'y_true': [], 'y_pred': []} for score_type in model_names}

    for line in lines:
        _, text, gold_genre = line

        for name, pred_genre in zip(model_names, model.predict_and_return_all(text)):
            scores[name]['y_true'].append(gold_genre)
            scores[name]['y_pred'].append(pred_genre)

    for name in model_names:
        print(f"Truth: {scores[name]['y_true']}")
        print(f"Predictions: {scores[name]['y_pred']}")
        f1 = f1_score(scores[name]['y_true'], scores[name]['y_pred'], average=average)
        print(f"F1-score for {name}: {f1}")

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