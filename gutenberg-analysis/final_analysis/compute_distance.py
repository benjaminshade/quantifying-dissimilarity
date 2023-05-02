'''
This code is for computing the quantity P(X<Y) for all measures except the JSD
(JSD is treated separately in final_analysis/optimal_alpha_new.py so that we can 
compute P(X<Y) for a range of alpha values).

Input:
On the command line, the user gives the name of the desired measure.

Output:
The output is a nested dictionary with the following structure:
{ 
    'new' : { 
            'author' : [ P(X < Y) for each of the 10 author subcorpora ] ,
            'subject' : [ P(X < Y) for each of the 10 subject subcorpora ] ,
            'time' : [ P(X < Y) for each of the 10 time period subcorpora ] ,
            }
    'new_controlled' : { 
            'author' : [ P(X < Y) for each of the 10 author subcorpora ] ,
            'subject' : [ P(X < Y) for each of the 10 subject subcorpora ] ,
            'time' : [ P(X < Y) for each of the 10 time period subcorpora ] ,
            }
}
The key 'new' refers to UNCONTROLLED subcorpora (i.e. we do not control for 
confounding between author, subject and time period), while 'new_controlled'
refers to CONTROLLED subcorpora. Refer to the paper for more details.
'''

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


# Extract the chosen measure from command line argument
measure_name = sys.argv[1]


'''
Below, we define the different dissimilarity measures we wish to investigate.
Users are encouraged to define additional measures here for their own research.

b1, b2 are word frequency distribtions.
vec1, vec2 are vector embeddings.
'''

# Jaccard distance between vocabularies
def jaccard_distance(b1, b2):
    b1_words = set(b1.keys())
    b2_words = set(b2.keys())
    union = b1_words.union(b2_words)
    intersection = b1_words.intersection(b2_words)
    return 1 - len(intersection)/len(union)

# Overlap dissimilarity between vocabularies
def overlap_distance(b1,b2):
    b1_words = set(b1.keys())
    b2_words = set(b2.keys())
    intersection = b1_words.intersection(b2_words)
    return 1 - len(intersection)/min(len(b1_words),len(b2_words))

# Relative difference in length between two texts
def length_distance(b1,b2):
    n1 = sum(b1.values())
    n2 = sum(b2.values())
    return abs(n1 - n2)/(n1 + n2)

# Euclidean distance between word frequency distributions
def euclidean_freq(b1, b2):
    vec1, vec2 = get_p12_same_support(b1, b2)
    return distance.minkowski(vec1, vec2, 2)

# Angular distance between two vectors
def angular_distance(vec1, vec2):
    cosine_sim = np.inner(vec1, vec2) / (norm(vec1) * norm(vec2))
    return np.arccos(cosine_sim)/np.pi

# Euclidean distance between two vectors
def euclidean_vec(vec1, vec2):
    return norm(vec1 - vec2)

# Minkowski distance with parameter p
def minkowski_distance(vec1, vec2, p):
    return distance.minkowski(vec1, vec2, p)

# Minkowski distance with parameter p, with vectors first normalised
def minkowski_distance_normed(vec1, vec2, p):
    return distance.minkowski(vec1/norm(vec1), vec2/norm(vec2), p)

# Jensen-Shannon divergence between embedding vectors
def jsd_emb(vec1, vec2, alpha=1):
    return D_alpha(vec1/sum(vec1), vec2/sum(vec2), alpha=alpha)
    

'''
This mapping dictionary retrieves necessary information based on the user's
input to the command line. The lists contain three elements:
    1. String indicating if the measure is considering vocabularies or word
       frequencies (freq) or vector embeddings (emb)
    2. The corresponding function for computing dissimilarity.
    3. Any additional parameters required for the function.
'''
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
    '''
    Computes P(X < Y) for the given dissimilarity measure (for texts represented by vocabularies 
    or word frequencies).

    Parameters
    ----------
    distance_measure : function
        Function for computing dissimilarity (list element 1 above).

    measure_name : string
        Name of chosen function (list element 2 above).
    
    p : integer
        Additional parameter if required (list element 3 above).

    Returns
    -------
     : dictionary
        Results are saved in a pickle file titled '{measure_name}_results_new.pickle',
        located in the output_files directory. 
        The structure of the object in the pickle file is explained at the top of
        this file.
    '''
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
    '''
    Computes P(X < Y) for the given dissimilarity measure (for texts represented by 
    vector embeddings).

    Parameters
    ----------
    distance_measure : function
        Function for computing dissimilarity (list element 1 above).

    measure_name : string
        Name of chosen function (list element 2 above).
    
    p : integer
        Additional parameter if required (list element 3 above).

    Returns
    -------
     : dictionary
        Results are saved in a pickle file titled '{measure_name}_embedding_results_new.pickle',
        located in the output_files directory. 
        The structure of the object in the pickle file is explained at the top of
        this file.
    '''
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


# This code runs the above functions with the provided user input
chosen_distance = distances_map[measure_name]
if chosen_distance[0] == 'freq':
    word_frequency_results(chosen_distance[1], measure_name, chosen_distance[2])
elif chosen_distance[0] == 'emb':
    embedding_results(chosen_distance[1], measure_name, chosen_distance[2])




