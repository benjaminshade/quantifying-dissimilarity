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
from data_io import get_book

import re
import random


# Accessing the metadata
sys.path.append(os.path.join(path_gutenberg,'src'))
from metaquery import meta_query
mq = meta_query(path=os.path.join(path_gutenberg,'metadata','metadata.csv'), filter_exist=False)

from itertools import combinations
from data_io import get_book, get_p12_same_support
from metric_eval import resample_book, prob_x_less_than_y_freq_new, prob_x_less_than_y_emb_new
from jsd import jsdalpha
from ent import D_alpha


# Extract the inputs
measure_name = sys.argv[1]

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

def length_distance(b1,b2):
    n1 = sum(b1.values())
    n2 = sum(b2.values())
    return abs(n1 - n2)/(n1 + n2)

def euclidean_freq(b1, b2):
    vec1, vec2 = get_p12_same_support(b1, b2)
    return distance.minkowski(vec1, vec2, 2)

# Cosine similarity
def cosine_similarity(vec1, vec2):
    return np.inner(vec1, vec2) / (norm(vec1) * norm(vec2))

# Angular distance
def angular_distance(vec1, vec2):
    cosine_sim = cosine_similarity(vec1, vec2)
    return np.arccos(cosine_sim)/np.pi

# Euclidean distance
def euclidean_vec(vec1, vec2):
    return norm(vec1 - vec2)

# Minkowski distance
def minkowski_distance(vec1, vec2, p):
    return distance.minkowski(vec1, vec2, p)

def minkowski_distance_normed(vec1, vec2, p):
    return distance.minkowski(vec1/norm(vec1), vec2/norm(vec2), p)

def jsd_emb(vec1, vec2, alpha=1):
    return D_alpha(vec1/sum(vec1), vec2/sum(vec2), alpha=alpha)
    


# Dictionary mapping name to function
distances_map = {'jaccard' : ['freq', jaccard_distance, None],
                 'overlap' : ['freq', overlap_distance, None],
                 'text_length' : ['freq', length_distance, None],
                 'euclidean_freq' : ['freq', euclidean_freq, None],
                 'angular' : ['emb', angular_distance, None],
                 'euclidean' : ['emb', minkowski_distance, 2],
                 'manhattan' : ['emb', minkowski_distance, 1], 
                 'euclidean_normed' : ['emb', minkowski_distance_normed,2],
                 'jsd' : ['emb', jsd_emb, None]}


def word_frequency_results(distance_measure, measure_name, p = None):
    results = {}
    for controlled in ['new', 'new_controlled']:
        controlled_dict = {}
        for task in ['author', 'subject', 'time']:
            task_numbers = [i for i in range(11,21)]
            task_probs = []
            input_file_path = f'../output_files/{task}_corpora_{controlled}.pickle'
            with open(input_file_path, 'rb') as f:
                task_pickle = pickle.load(f)

            for task_num in task_numbers:
                pair_dict = task_pickle[task_num-11]
                prob = prob_x_less_than_y_freq_new(pair_dict, distance_measure, add_parameter = p)
                task_probs.append(prob)

                print(f'{task_num} : {prob},', sep=' ', end=' ', flush=True)

            controlled_dict[task] = task_probs.copy()
        results[controlled] = controlled_dict
        

    # Save the results
    output_file_path = f'../output_files/{measure_name}_results_new.pickle'
    with open(output_file_path, 'wb') as f:
        pickle.dump(results, f)


def embedding_results(distance_measure, measure_name, p = None):
    results = {}
    for controlled in ['new', 'new_controlled']:
        controlled_dict = {}
        for task in ['author', 'subject', 'time']:
            task_numbers = [i for i in range(11,21)]
            task_probs = []
            input_file_path = f'../output_files/{task}_corpora_{controlled}.pickle'
            with open(input_file_path, 'rb') as f:
                task_pickle = pickle.load(f)

            for task_num in task_numbers:
                pair_dict = task_pickle[task_num-11] 
                prob = prob_x_less_than_y_emb_new(pair_dict, distance_measure, add_parameter = p)
                task_probs.append(prob)

                print(f'{task_num} : {prob},', sep=' ', end=' ', flush=True)

            controlled_dict[task] = task_probs.copy()
        results[controlled] = controlled_dict
        
    # Save the results
    output_file_path = f'../output_files/{measure_name}_embedding_results_new.pickle'
    with open(output_file_path, 'wb') as f:
        pickle.dump(results, f)



chosen_distance = distances_map[measure_name]
if chosen_distance[0] == 'freq':
    word_frequency_results(chosen_distance[1], measure_name, chosen_distance[2])
elif chosen_distance[0] == 'emb':
    embedding_results(chosen_distance[1], measure_name, chosen_distance[2])




