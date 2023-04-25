# Importing relevant packages
import numpy as np
import pandas as pd
import os, sys
import pickle

from collections import Counter
import matplotlib.pyplot as plt

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
from metric_eval import esample_book, prob_x_less_than_y_freq_new, optimal_alpha_test_freq_new
from jsd import jsdalpha


# Extract the inputs
if sys.argv[1] == 'new':
    suffix = 'new'
elif sys.argv[1] == 'controlled':
    suffix = 'new_controlled'
task = sys.argv[2]
task_num = int(sys.argv[3])

if len(sys.argv) == 5:
    additional_arg = sys.argv[4]

# Retrieve the relevant pickle file
input_file_path = f'../output_files/{task}_corpora_{suffix}.pickle'

with open(input_file_path, 'rb') as f:
    task_pickle = pickle.load(f)

# Numbers are from 11-20
pair_dict = task_pickle[task_num-11]

# Chosen alpha values (values from 0 to 2 at intervals of length 0.05)
alphas = [i/20 for i in range(41)]

# Perform test on all 10 subcorpora, and store in a dictionary of lists
if len(sys.argv) == 5:
    if additional_arg == 'resample':
        sizes = [i*5000 for i in range(1,21)]
        final_result = {}
        for resample_size in sizes:
            current_result = optimal_alpha_test_freq_new(pair_dict, alphas, resample_size = resample_size)
            final_result[resample_size] = current_result
        output_file_path = f'../output_files/optimal_alpha_resample_{suffix}_{task}{task_num}.pickle'

    elif additional_arg == 'weights':
        final_result = optimal_alpha_test_freq_new(pair_dict, alphas, weights = True)
        output_file_path = f'../output_files/optimal_alpha_weights_{suffix}_{task}{task_num}.pickle'
else:
    final_result = optimal_alpha_test_freq_new(pair_dict, alphas)
    output_file_path = f'../output_files/optimal_alpha_{suffix}_{task}{task_num}.pickle'


# Save the results
with open(output_file_path, 'wb') as f:
    pickle.dump(final_result, f)