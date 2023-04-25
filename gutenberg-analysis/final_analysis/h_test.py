# Importing relevant packages
import numpy as np
import pandas as pd
import os, sys
import pickle

from collections import Counter
import matplotlib.pyplot as plt

from numpy.linalg import norm
from scipy.spatial.distance import cdist
from scipy.spatial import distance


# %matplotlib inline

# %load_ext autoreload
# %autoreload 2

## path to the downloaded gutenberg corpus
path_gutenberg = os.path.join(os.pardir,os.pardir,'gutenberg')

## import internal helper functions
src_dir = os.path.join(os.pardir,'src')
sys.path.append(src_dir)

import re
import random


# Accessing the metadata
sys.path.append(os.path.join(path_gutenberg,'src'))
from metaquery import meta_query
mq = meta_query(path=os.path.join(path_gutenberg,'metadata','metadata.csv'), filter_exist=False)

from itertools import combinations
from data_io import get_book, get_p12_same_support
from metric_eval import resample_book
from jsd import jsdalpha
from ent import D_alpha


# Extract the inputs
size = sys.argv[1] # size is 'small' or 'large'
measure_name = sys.argv[2]

if size == 'small':
    N = 500
    h_vals = [i for i in range(1,21)]
elif size == 'large':
    N = 10000
    h_vals = [i for i in range(1,11)]

# Define relevant metrics
def jaccard_distance(b1, b2):
    b1_words = set(b1.keys())
    b2_words = set(b2.keys())
    union = b1_words.union(b2_words)
    intersection = b1_words.intersection(b2_words)
    return 1 - len(intersection)/len(union)

def overlap_distance(b1,b2):
    b1_words = set(b1.keys())
    b2_words = set(b2.keys())
    intersection = b1_words.intersection(b2_words)
    return 1 - len(intersection)/min(len(b1_words),len(b2_words))

# Cosine similarity
def cosine_similarity(vec1, vec2):
    return np.inner(vec1, vec2) / (norm(vec1) * norm(vec2))

# Angular distance
def angular_distance(vec1, vec2):
    cosine_sim = cosine_similarity(vec1, vec2)
    return np.arccos(cosine_sim)/np.pi

# Manhattan distance
def manhattan_distance(vec1, vec2):
    return distance.minkowski(vec1, vec2, 1)

# +-------------------------------------------------------------------------------
# Frequencies
# +-------------------------------------------------------------------------------

# Redefine our metric eval function
def prob_x_less_than_y_freq_h(pair_dict, metric, N, h, add_parameter=None):
    same = []
    same_pairs = pair_dict['same']
    for pair in same_pairs:
        book1 = get_book(pair[0])
        book2 = get_book(pair[1])

        book1 = resample_book(book1, N)
        book2 = resample_book(book2, N*h)

        if add_parameter is not None:
            distance = metric(book1, book2, add_parameter)
        else:
            distance = metric(book1, book2)
        same.append(distance)
    
    different = []
    different_pairs = pair_dict['different']
    for pair in different_pairs:
        book1 = get_book(pair[0])
        book2 = get_book(pair[1])

        book1 = resample_book(book1, N)
        book2 = resample_book(book2, N*h)

        if add_parameter is not None:
            distance = metric(book1, book2, add_parameter)
        else:
            distance = metric(book1, book2)
        different.append(distance)
    
    # Pairing all same-author scores with all different-author scores
    differences = []
    for score1 in same:
        for score2 in different:
            current_diff = score1 - score2
            differences.append(current_diff)
    
    # Compute the probability that this difference is less than 0 
    # Since we're comparing distance measures, a negative difference means the same-author
    # books are closer together than the different-author books
    processed_diffs = [1 if i < 0 else 0 for i in differences]
    probability = sum(processed_diffs)/len(processed_diffs)
    
    return probability


# +-------------------------------------------------------------------------------
# Embeddings
# +-------------------------------------------------------------------------------

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
max_length = 206

# Function for creating word embeddings from strings and averaging them
def embed__and_combine_strings(sentences):
    embeddings = model.encode(sentences)
    if len(embeddings) == 0:
        return None
    return sum(embeddings)/len(embeddings)

# Function for embedding a tokenized book
def embed_book_tokens(book):
    stored_strings = []
    stored_lengths = []

    i = 0
    while i < len(book):
        if len(book) - i >= max_length:
            emb_input = book[i:i + max_length]
        else:
            emb_input = book[i:len(book)]
    
        # Past the words together
        emb_input_str = " ".join(emb_input)

        # Store input strings
        stored_strings.append(emb_input_str)
        stored_lengths.append(len(emb_input))

        i += max_length

    embedded_book = embed__and_combine_strings(stored_strings)

    return embedded_book


def prob_x_less_than_y_emb_h(pair_dict, metric, N, h, add_parameter=None):
    # Load in the embedding dictionary
    # input_file_path = f'../output_files/pretrained_{model_name}_all.pickle'
    # with open(input_file_path, 'rb') as f:
    #     embedding_dict = pickle.load(f)
    same = []
    same_pairs = pair_dict['same']
    for pair in same_pairs:
        book1_full = get_book(pair[0], level='tokens')
        book2_full = get_book(pair[1], level='tokens')
        
        book1_median = int(len(book1_full)/3)
        book2_median = int(len(book2_full)/3)
        try:
            book1_short = book1_full[book1_median:book1_median+N] 
            book2_short = book2_full[book2_median:book2_median+N*h]
        except:
            continue

        book1 = embed_book_tokens(book1_short)
        book2 = embed_book_tokens(book2_short)

        try:
            distance = metric(book1, book2)
            same.append(distance)
        except:
            continue
    
    different = []
    different_pairs = pair_dict['different']
    for pair in different_pairs:
        book1_full = get_book(pair[0], level='tokens')
        book2_full = get_book(pair[1], level='tokens')
        
        book1_median = int(len(book1_full)/3)
        book2_median = int(len(book2_full)/3)
        try:
            book1_short = book1_full[book1_median:book1_median+N] 
            book2_short = book2_full[book2_median:book2_median+N*h]
        except:
            continue

        book1 = embed_book_tokens(book1_short)
        book2 = embed_book_tokens(book2_short)

        try:
            distance = metric(book1, book2)
            different.append(distance)
        except:
            continue
    
    # Pairing all same-author scores with all different-author scores
    differences = []
    for score1 in same:
        for score2 in different:
            current_diff = score1 - score2
            differences.append(current_diff)
    
    # Compute the probability that this difference is less than 0 
    # Since we're comparing distance measures, a negative difference means the same-author
    # books are closer together than the different-author books
    processed_diffs = [1 if i < 0 else 0 for i in differences]
    probability = sum(processed_diffs)/len(processed_diffs)
    
    return probability



# +-------------------------------------------------------------------------------
# All
# +-------------------------------------------------------------------------------

def resample_results(freq_emb, distance_measure, measure_name, N, add_parameter = None):
    if freq_emb == 'emb':
        num_pairs = 60
        pxy_func = prob_x_less_than_y_emb_h
    elif freq_emb == 'freq':
        num_pairs = 50
        pxy_func = prob_x_less_than_y_freq_h
    
    results = {}
    for task in ['author', 'subject', 'time']:
        print(f'{task}: ', sep=' ', end=' ', flush=True)
        task_dict = {}
        input_file_path = f'../output_files/{task}_corpora_new.pickle'
        with open(input_file_path, 'rb') as f:
            task_pickle = pickle.load(f)
        for h in h_vals:
            resample_probs = []
            task_numbers = [i for i in range(11,21)]
            for task_num in task_numbers:
                pair_dict = task_pickle[task_num-11]
                # Only take first num_pairs books
                pair_dict_small = {}
                pair_dict_small['same'] = pair_dict['same'][0:num_pairs]
                pair_dict_small['different'] = pair_dict['different'][0:num_pairs]
                if measure_name == 'jsdopt':
                    prob = pxy_func(pair_dict_small, distance_measure, N, h, add_parameter = add_parameter[task])
                elif measure_name == 'embedding':
                    prob = pxy_func(pair_dict_small, distance_measure[task], N, h, add_parameter = None)
                else:
                    prob = pxy_func(pair_dict_small, distance_measure, N, h, add_parameter = None)

                resample_probs.append(prob)

            print(f'{h}, {resample_probs}', sep=' ', end=' ', flush=True)

            task_dict[h] = resample_probs.copy()
        results[task] = task_dict

    # Save the results
    output_file_path = f'../output_files/{measure_name}_h_results_{size}.pickle'
    with open(output_file_path, 'wb') as f:
        pickle.dump(results, f)
    


# Dictionary mapping name to function
optimal_alphas = {'author':0.65, 'subject':0.6, 'time':0.8}
embedding_metrics = {'author':angular_distance, 'subject':angular_distance, 'time':manhattan_distance}
distances_map = {'jaccard' : ['freq', jaccard_distance, None],
                 'overlap' : ['freq', overlap_distance, None],
                 'jsd1' : ['freq', jsdalpha, None],
                 'jsdopt' : ['freq', jsdalpha, optimal_alphas],
                 'embedding' : ['emb', embedding_metrics, None],}
                 

parameters = distances_map[measure_name]
resample_results(parameters[0], parameters[1], measure_name, N, add_parameter = parameters[2])




