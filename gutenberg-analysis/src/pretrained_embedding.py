'''
This code is for generating vector embeddings of all books in the corpus,
and then saving these embeddings in a dictionary. It is faster to embed all
the books in one go and then just refer to this dictionary, rather than embed 
them each time the quantity P(X<Y) is computed.

If you wish to use a different embedding model, load and use it in this file.

Input:
One the command line, the user gives the following inputs:
    1. Model number, selects chosen model from list provided below.
    2. Start index, which book is the corpus to begin with. When we tried to
       embed the corpus in one go, we encountered computational issues. Thus,
       we resolved to embed the books in batches of 1000, and then combine these
       all into one dictionary afterwards.

Output:
A dictionary containing 1000 embedded books with the structure
    { book : vector embedding }
This dictionary is stored in a pickle file (in the output_files directory) 
that has name 'pretrained_{model_name}_{start_index}.pickle'
    model_name : name of the chosen embedding model
    start_index : which book in the corpus we began with
'''

from sentence_transformers import SentenceTransformer

# Importing relevant packages
import numpy as np
import pandas as pd
import os, sys
import pickle

from collections import Counter
import matplotlib.pyplot as plt


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
from jsd import jsdalpha


# Extract the inputs
model_num = int(sys.argv[1])
start_index = int(sys.argv[3])

'''
Specification of models. Please add new models here.

Note: max_length refers to the maximum sequence length that we input into the 
SentenceBERT model. This length is specified in their documentation. However, the
SentenceBERT model performs an additional tokenisation step that often results in our
tokens being separated further, inadvertently increasing the number of tokens being 
input. Thus, we (somewhat arbitrarily) reduce the input length so that we truncate
as few words as possible (while still having a large enough input length).
'''
if model_num == 1:
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    model_name = 'all-mpnet-base-v2'
    max_length = 384 - 60
elif model_num == 2:
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    model_name = 'all-MiniLM-L6-v2'
    max_length = 256 - 50
elif model_num == 3:
    model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')
    model_name = 'all-distilroberta-v1'
    max_length = 512 - 80


def embed_book_tokens(book, model, max_length):
    '''
    Generates an embedding of the given book.

    Parameters
    ----------
    book : list
        A book with level = 'tokens'. This is a list of words, in the order they appear 
        in the original text.
    
    model : embedding model
        The model used to embed the sentences in the book.
    
    max_length : integer
        The maximum number of tokens being input to the embedding model at once.
    
    Returns
    -------
     : numpy array
        The embedded book, represented by a numpy array.
    '''
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

    embeddings = model.encode(stored_strings)
    if len(embeddings) == 0:
        embedded_book = None
    else:
        embedded_book = sum(embeddings)/len(embeddings)

    return embedded_book


# Perform necessary filtering
mq.reset()
mq.filter_lang('en',how='only')
# Only select books with more than 20 downloads
df = mq.get_df()
mq.df = df[df['downloads'] >= 20]
# 1800 onwards
mq.filter_year([1800, 2050])
# Filter out data with no subject listed
df = mq.get_df()
mq.df = df[df['subjects'] != 'set()']
# Filter out entries that don't have author birth or death year
df = mq.get_df()
mq.df = df[df[['authoryearofbirth', 'authoryearofdeath']].notnull().all(1)]


# Iteratively embed all books
ids = mq.get_ids()
if start_index == 13000: # There are <14k books in our filtered corpus, so index to 14k gives an error
    all_ids = ids[start_index:]
else:
    all_ids = ids[start_index:start_index + 1000]
total = len(all_ids)
print(total)

final_result = {}

i = 0
for id in all_ids:
    try:
        book = get_book(id, level='tokens')
    except:
        continue
    book_emb = embed_book_tokens(book, model)

    final_result[id] = book_emb

    i += 1
    percent_complete = round(100*i/total,2)

    if percent_complete%10 <= 0.1:
        print(percent_complete)

# Save the results
output_file_path = f'../output_files/pretrained_{model_name}_{start_index}.pickle'
with open(output_file_path, 'wb') as f:
    pickle.dump(final_result, f)