'''
This code is for computing P(X<Y) after resampling all texts to specified 
lengths. This is for texts of equal lengths, see final_analysis/h_test.py 
for using texts of unequal lengths.

Input:
One the command line, the user gives the following inputs:
    1. The order of magnitude of the length of the resampled text. 
       This can be 'very_small', 'small' or 'large'.
       - very_small : between 10 and 1,000
       - small : between 1,000 and 10,000
       - large : between 10,000 and 100,000
    2. The name of the desired measure.
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
from pretrained_embedding import embed_book_tokens


# Extract the inputs
size = sys.argv[1]
measure_name = sys.argv[2]

# Specific resample lengths for each size. Alter these to suit your purposes.
if size == 'very_small':
    resample_sizes = [10*i for i in range(1,10)] + [i*100 for i in range(1,11)]
elif size == 'small':
    resample_sizes = [500*i for i in range(1,21)]
elif size == 'large':
    resample_sizes = [18000, 32000, 56000, 100000]

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

# Angular distance
def angular_distance(vec1, vec2):
    cosine_sim = np.inner(vec1, vec2) / (norm(vec1) * norm(vec2))
    return np.arccos(cosine_sim)/np.pi

# Manhattan distance between two vectors
def manhattan_distance(vec1, vec2):
    return distance.minkowski(vec1, vec2, 1)

# +-------------------------------------------------------------------------------
# Frequencies
# +-------------------------------------------------------------------------------


def prob_x_less_than_y_freq_n(pair_dict, metric, resample_size, add_parameter=None):
    '''
    Computes P(X < Y) for texts represented by either vocabularies or word frequency distributions, 
    but first resamples all texts to a given length. 
    Refer to paper for more details on the quantity P(X < Y).

    Parameters
    ----------
    pair_dict : dictionary
        Dictionary with structure { 'same' : [ list of book pairs ( book1, book2 ) ],
                                    'different' : [ list of book pairs ( book1, book2 ) ] }
        In the 'same' list, book1 and book2 have the same author/subject/time period, while in the 'different'
        list, they are from different groups.

    metric : function 
        Function for computing dissimilarity between texts represented either by vocabularies
        or by word frequencies.
    
    resample_size : integer
        The desired length of the resampled texts.
    
    OPTIONAL:
    add_parameter : varies
        Additional parameter for metric.

    Returns
    -------
     : float
        Float value representing the quantity P(X < Y).
    '''
    same = []
    same_pairs = pair_dict['same']
    for pair in same_pairs:
        book1 = get_book(pair[0])
        book2 = get_book(pair[1])

        book1 = resample_book(book1, resample_size)
        book2 = resample_book(book2, resample_size)

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

        book1 = resample_book(book1, resample_size)
        book2 = resample_book(book2, resample_size)

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


def prob_x_less_than_y_emb_n(pair_dict, metric, resample_size, add_parameter=None):
    '''
    Computes P(X < Y) for texts represented by dense vector embeddings, but first resamples 
    all texts to a given length. 
    Refer to paper for more details on the quantity P(X < Y).. 

    Parameters
    ----------
    pair_dict : dictionary
        Dictionary with structure { 'same' : [ list of book pairs ( book1, book2 ) ],
                                    'different' : [ list of book pairs ( book1, book2 ) ] }
        In the 'same' list, book1 and book2 have the same author/subject/time period, while in the 'different'
        list, they are from different groups.

    metric : function 
        Function for computing dissimilarity between texts represented vector embeddings. 
    
    resample_size : integer
        The desired length of the resampled texts.
    
    OPTIONAL:
    add_parameter : varies
        Additional parameter for metric.

    Returns
    -------
     : float
        Float value representing the quantity P(X < Y).
    '''
    same = []
    same_pairs = pair_dict['same']
    for pair in same_pairs:
        book1_full = get_book(pair[0], level='tokens')
        book2_full = get_book(pair[1], level='tokens')
        
        # To get the book to the desired length, we start one third of the way 
        # through, and then take the next N or Nh words. 
        book1_third = int(len(book1_full)/3)
        book2_third = int(len(book2_full)/3)
        try:
            book1_short = book1_full[book1_third:book1_third+resample_size] 
            book2_short = book2_full[book2_third:book2_third+resample_size]
        except:
            continue

        book1 = embed_book_tokens(book1_short, model, max_length)
        book2 = embed_book_tokens(book2_short, model, max_length)
        
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
        
        # To get the book to the desired length, we start one third of the way 
        # through, and then take the next N or Nh words. 
        book1_third = int(len(book1_full)/3)
        book2_third = int(len(book2_full)/3)
        try:
            book1_short = book1_full[book1_third:book1_third+resample_size] 
            book2_short = book2_full[book2_third:book2_third+resample_size]
        except:
            continue

        book1 = embed_book_tokens(book1_short, model, max_length)
        book2 = embed_book_tokens(book2_short, model, max_length)

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

def resample_results(freq_emb, distance_measure, measure_name, add_parameter = None):
    '''
    Computes resampling results for all groups (author, subject and time) for all
    uncontrolled subcorpora.

    Parameters
    ----------
    freq_emb : string
        Indicates if the chosen measure is considering vocabularies or word frequencies ('freq') 
        or vector embeddings ('emb').
    
    distance_measure : function
        Function for computing dissimilarity between texts.

    measure_name : string
        The name of the chosen dissimilarity measure.

    
    OPTIONAL:
    add_parameter : varies
        Additional parameter for metric.

    Returns
    -------
    Nothing is returned, but a dictionary is saved as a pickle file.
    The dictionary has the following structure:
    {
        'author' : { resample_size1 : [ list of P(X<Y) for 10 subcorpora ] ,
                        resample_size2 : [ list of P(X<Y) for 10 subcorpora ] ,
                        .... } ,
        'subject' : { resample_size1 : [ list of P(X<Y) for 10 subcorpora ] ,
                        resample_size2 : [ list of P(X<Y) for 10 subcorpora ] ,
                        .... } ,
        'time' : { resample_size1 : [ list of P(X<Y) for 10 subcorpora ] ,
                    resample_size2 : [ list of P(X<Y) for 10 subcorpora ] ,
                        .... } ,
    }
    '''
    if freq_emb == 'emb':
        num_pairs = 60 # Can reduce number of pairs from 1000 to X to reduce computation time
        pxy_func = prob_x_less_than_y_emb_n
    elif freq_emb == 'freq':
        num_pairs = 100
        pxy_func = prob_x_less_than_y_freq_n
    
    results = {}
    for task in ['author', 'subject', 'time']:
        print(f'{task}: ', sep=' ', end=' ', flush=True)
        task_dict = {}
        input_file_path = f'../output_files/{task}_corpora_new.pickle'
        with open(input_file_path, 'rb') as f:
            task_pickle = pickle.load(f)
        for resample_size in resample_sizes:
            resample_probs = []
            task_numbers = [i for i in range(11,21)]
            for task_num in task_numbers:
                pair_dict = task_pickle[task_num-11]
                # Only take first n number of pairs of books
                pair_dict_small = {}
                pair_dict_small['same'] = pair_dict['same'][0:num_pairs]
                pair_dict_small['different'] = pair_dict['different'][0:num_pairs]
                if measure_name == 'jsdopt':
                    prob = pxy_func(pair_dict_small, distance_measure, resample_size, add_parameter = add_parameter[task])
                elif measure_name == 'embedding':
                    prob = pxy_func(pair_dict_small, distance_measure[task], resample_size, add_parameter = None)
                else:
                    prob = pxy_func(pair_dict_small, distance_measure, resample_size, add_parameter = None)

                resample_probs.append(prob)

            print(f'{resample_size}, {resample_probs}', sep=' ', end=' ', flush=True)

            task_dict[resample_size] = resample_probs.copy()
        results[task] = task_dict

    # Save the results
    output_file_path = f'../output_files/{measure_name}_final_resample_results_{size}.pickle'
    with open(output_file_path, 'wb') as f:
        pickle.dump(results, f)
    

'''
This mapping dictionary retrieves necessary information based on the user's
input to the command line. The lists contain three elements:
    1. String indicating if the measure is considering vocabularies or word
       frequencies (freq) or vector embeddings (emb)
    2. The corresponding function for computing dissimilarity.
    3. Any additional parameters required for the function.
'''
optimal_alphas = {'author':0.65, 'subject':0.6, 'time':0.8}
embedding_metrics = {'author':angular_distance, 'subject':angular_distance, 'time':manhattan_distance} # Optimal embedding metrics
distances_map = {'jaccard' : ['freq', jaccard_distance, None],
                 'overlap' : ['freq', overlap_distance, None],
                 'jsd1' : ['freq', jsdalpha, None],
                 'jsdopt' : ['freq', jsdalpha, optimal_alphas],
                 'embedding' : ['emb', embedding_metrics, None],}
                 

# Generate the results
parameters = distances_map[measure_name]
resample_results(parameters[0], parameters[1], measure_name, add_parameter = parameters[2])

