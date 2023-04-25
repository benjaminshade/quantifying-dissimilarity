"""Functions to handle data I/O."""

import numpy as np
import os, sys
from metaquery import meta_query
import random

def get_book(pg_id, path_gutenberg = None, level = 'counts'):
    '''
    Retrieve the data from a single book.
    pg_id, str: id of book in format 'PG12345'

    OPTIONAL:
    
    path_gutenberg, str: location of directory of gutenberg data
        default ../../gutenberg/


    level, which granularity
        - 'counts', dict(word,count) [default]
        - 'tokens', list of tokens (str)
        - 'text', single str

    '''

    ## location of the gutenberg data
    if path_gutenberg == None:
        path_gutenberg = os.path.join(os.pardir,os.pardir,'gutenberg')
    if level == 'counts':
        ## counts -- returns a dictionary {str:int}
        path_read = os.path.join(path_gutenberg,'data','counts')
        fname_read = '%s_counts.txt'%(pg_id)
        filename = os.path.join(path_read,fname_read)
        with open(filename,'r') as f:
            x = f.readlines()

        words = [h.split()[0] for h in x]
        counts = [int(h.split()[1]) for h in x]
        dict_word_count = dict(zip(words,counts))
        return dict_word_count

    elif level == 'tokens':
        ## tokens --> returns a list of strings 
        path_read = os.path.join(path_gutenberg,'data','tokens')
        fname_read = '%s_tokens.txt'%(pg_id)
        filename = os.path.join(path_read,fname_read)
        with open(filename,'r') as f:
            x = f.readlines()
        list_tokens = [h.strip() for h in x]
        return list_tokens

    elif level == 'text':
        ## text --> returns a string 
        path_read = os.path.join(path_gutenberg,'data','text')
        fname_read = '%s_text.txt'%(pg_id)
        filename = os.path.join(path_read,fname_read)
        with open(filename,'r') as f:
            x = f.readlines()
        text =  ' '.join([h.strip() for h in x])
        return text

    else:
        print('ERROR: UNKNOWN LEVEL')
        return None

def get_dict_words_counts(filename):
    """
    Read a file and make a dictionary with words and counts.

    Parameters
    ----------
    filename : str
        Path to file.

    Returns
    -------
     : dict
        Dictionary with words in keys and counts in values.

    """
    with open(filename, 'r') as f:
        x = f.readlines()
    if x[0] == '\n':
        # an empty book
        words = []
        counts = []
    else:
        words = [h.split()[0] for h in x]
        counts = [int(h.split()[1]) for h in x]
    return dict(zip(words, counts))


def get_p12_same_support(
        dict_wc1,
        dict_wc2):
    """
    Get probabilities with common support.

    For two dictionaries of the form {word:count},
    make two arrays p1 and p2 holding probabilites
    in which the two distributions have the same support.

    Parameters
    -----------
    dict_wc1, dict_wc2 : dict
        Dictionaries of the form {word: count}.

    Returns
    -------
    arr_p1, arr_p2 : np.array (float)
        Normalized probabilites with common support.

    """
    N1 = sum(list(dict_wc1.values()))
    N2 = sum(list(dict_wc2.values()))
    # union of all words sorted alphabetically
    words1 = list(dict_wc1.keys())
    words2 = list(dict_wc2.keys())
    words_12 = sorted(list(set(words1).union(set(words2))))
    V = len(words_12)
    arr_p1 = np.zeros(V)
    arr_p2 = np.zeros(V)
    for i_w, w in enumerate(words_12):
        try:
            arr_p1[i_w] = dict_wc1[w]/N1
        except KeyError:
            pass
        try:
            arr_p2[i_w] = dict_wc2[w]/N2
        except KeyError:
            pass
    return arr_p1, arr_p2


# Retrieve all books in a dictionary of structure
# { author/subject : [list of book ids] }
def get_all_books(ids_dict):
    key_list = list(ids_dict.keys()).copy()

    books_dict = {}

    for key in key_list:
        new_book_list = []

        ids = ids_dict[key]
        for id in ids:
            try:
                new_book = get_book(id)
                new_book_list.append(new_book)
            except:
                continue
        
        books_dict[key] = new_book_list.copy()
    
    return books_dict


# Retrieve all books in a list
def get_all_books_from_list(ids_list):
    books_not_found = 0
    new_book_list = []
    for id in ids_list:
        try:
            new_book = get_book(id)
            new_book_list.append(new_book)
        except:
            books_not_found += 1
            continue
    return new_book_list