from sentence_transformers import SentenceTransformer

# Load word tokenizer
# import nltk
# nltk.download('punkt')
# from nltk.tokenize import word_tokenize, sent_tokenize

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
from jsd import jsdalpha


# Extract the inputs
model_num = int(sys.argv[1])
task = sys.argv[2]
start_index = int(sys.argv[3])

# Specify the model
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

# # Load the model
# model_file_path = f'../output_files/{model_name}.pickle'
# with open(model_file_path, 'rb') as f:

# If we wish to embed all books in the corpus
if task == 'all':
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

    ids = mq.get_ids()
    if start_index == 13000:
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
        book_emb = embed_book_tokens(book)

        final_result[id] = book_emb

        i += 1
        percent_complete = round(100*i/total,2)

        if percent_complete%10 <= 0.1:
            print(percent_complete)

    # Save the results
    output_file_path = f'../output_files/pretrained_{model_name}_{start_index}.pickle'
    with open(output_file_path, 'wb') as f:
        pickle.dump(final_result, f)

else:
    task_num = int(sys.argv[3])
    # Retrieve the relevant pickle file
    task_pickle = []
    input_file_path = f'../output_files/{task}_corpora.pickle'

    with open(input_file_path, 'rb') as f:
        while True:
            try:
                task_pickle.append(pickle.load(f))
            except EOFError:
                break

    ids_dict = task_pickle[task_num-1]


    # Code for embedding book tokens
    final_result = {}

    for key in ids_dict:
        embedded_books = []
        
        ids = ids_dict[key]
        
        for id in ids:
            try:
                book = get_book(id, level='tokens')
            except:
                continue
            book_emb = embed_book_tokens(book)

            embedded_books.append(book_emb)

        final_result[key] = embedded_books.copy()

        print(key, 'has been completed!')



    # Save the results
    output_file_path = f'../output_files/pretrained_{model_name}_{task}{task_num}.pickle'
    with open(output_file_path, 'wb') as f:
        pickle.dump(final_result, f)