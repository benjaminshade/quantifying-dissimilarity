'''The following functions are used in our evaluation of various distance metrics'''
from data_io import get_book
from jsd import jsdalpha
from itertools import combinations
import random
import pickle


def resample_book(book, size):
    '''
    Generates a resampling of the word frequency distribution of a given book.

    Parameters
    ----------
    book : dictionary
        A book with level = 'counts'. This has the structure { word : frequency }
    
    size : integer
        The desired length of the resampled book (i.e. the total number of words)

    Returns
    -------
     : dict
        Dictionary of same structure { word : frequency }, but with resampled
        frequencies.
    '''
    vocab = []
    weights = []
    for word in book:
        vocab.append(word)
        weights.append(book[word])

    new_book = {}

    samp = random.choices(vocab, weights, k = size)

    for word in samp:
        if word in new_book:
            new_book[word] += 1
        else:
            new_book[word] = 1

    return new_book


def prob_x_less_than_y_freq_new(pair_dict, metric, add_parameter=None, resample_size=None, weights=False):
    '''
    Computes P(X < Y) for texts represented by either vocabularies or word frequency distributions. 
    Refer to paper for more details on this quantity.

    Parameters
    ----------
    pair_dict : dictionary
        Dictionary with structure { 'same' : [ list of book pairs ( book1, book2 ) ],
                                    'different' : [ list of book pairs ( book1, book2 ) ] }
        In the 'same' list, book1 and book2 have the same author/subject/time period, while in the 'different'
        list, they are from different groups.

    metric : function 
        Function for computing dissimilarity between texts represented either by vocabularies
        or by word frequencies. See final_analysis/compute_distance.py for examples.
    
    OPTIONAL:
    add_parameter : varies
        Additional parameter for metric.
    
    resample_size : integer
        Desired resample size (leave as None to use original texts).
    
    weights : boolean
        Relevant only to JSD, determines how to weight each distribution. See 'weights' parameter 
        in src/jsd.py to observe how it is used.

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

        if resample_size is not None:
            book1 = resample_book(book1, resample_size)
            book2 = resample_book(book2, resample_size)

        if add_parameter is not None:
            distance = metric(book1, book2, add_parameter, weights=weights)
        else:
            distance = metric(book1, book2)
        same.append(distance)
    
    different = []
    different_pairs = pair_dict['different']
    for pair in different_pairs:
        book1 = get_book(pair[0])
        book2 = get_book(pair[1])

        if resample_size is not None:
            book1 = resample_book(book1, resample_size)
            book2 = resample_book(book2, resample_size)

        if add_parameter is not None:
            distance = metric(book1, book2, add_parameter, weights=weights)
        else:
            distance = metric(book1, book2)
        different.append(distance)
    
    # Pairing all same-group scores with all different-group scores
    differences = []
    for score1 in same:
        for score2 in different:
            current_diff = score1 - score2
            differences.append(current_diff)
    
    # Compute the probability that this difference is less than 0 
    # Since we're comparing distance measures, a negative difference means the same-group
    # books are closer together than the different-group books
    processed_diffs = [1 if i < 0 else 0 for i in differences]
    probability = sum(processed_diffs)/len(processed_diffs)
    
    return probability


def prob_x_less_than_y_emb_new(pair_dict, metric, add_parameter=None, model_name = 'all-MiniLM-L6-v2'):
    '''
    Computes P(X < Y) for texts represented by dense vector embeddings. 
    Refer to paper for more details on this quantity.

    Parameters
    ----------
    pair_dict : dictionary
        Dictionary with structure { 'same' : [ list of book pairs ( book1, book2 ) ],
                                    'different' : [ list of book pairs ( book1, book2 ) ] }
        In the 'same' list, book1 and book2 have the same author/subject/time period, while in the 'different'
        list, they are from different groups.

    metric : function 
        Function for computing dissimilarity between texts represented vector embeddings. 
        See final_analysis/compute_distance.py for examples.
    
    OPTIONAL:
    add_parameter : varies
        Additional parameter for metric.
    
    model_name : str
        The name of the desired embedding model.

    Returns
    -------
     : float
        Float value representing the quantity P(X < Y).
    '''
    # Load in the embedding dictionary
    input_file_path = f'../output_files/pretrained_{model_name}_all.pickle'
    with open(input_file_path, 'rb') as f:
        embedding_dict = pickle.load(f)
    
    same = []
    same_pairs = pair_dict['same']
    for pair in same_pairs:
        book1 = embedding_dict[pair[0]]
        book2 = embedding_dict[pair[1]]

        if (book1 is None) or (book2 is None):
            continue

        if add_parameter is not None:
            distance = metric(book1, book2, add_parameter)
        else:
            distance = metric(book1, book2)
        same.append(distance)
    
    different = []
    different_pairs = pair_dict['different']
    for pair in different_pairs:
        book1 = embedding_dict[pair[0]]
        book2 = embedding_dict[pair[1]]

        if (book1 is None) or (book2 is None):
            continue

        if add_parameter is not None:
            distance = metric(book1, book2, add_parameter)
        else:
            distance = metric(book1, book2)
        different.append(distance)
    
    # Pairing all same-group scores with all different-group scores
    differences = []
    for score1 in same:
        for score2 in different:
            current_diff = score1 - score2
            differences.append(current_diff)
    
    # Compute the probability that this difference is less than 0 
    # Since we're comparing distance measures, a negative difference means the same-group
    # books are closer together than the different-group books
    processed_diffs = [1 if i < 0 else 0 for i in differences]
    probability = sum(processed_diffs)/len(processed_diffs)
    
    return probability


# This function outputs a list, where each value is P(X < Y) for a particular alpha
def optimal_alpha_test_freq_new(pair_dict, alphas, resample_size = None, weights = False, metric = jsdalpha):
    '''
    Computes P(X < Y) for the given values of the alpha parameter (only relevent to JSD).
    Refer to paper for more details.

    Parameters
    ----------
    pair_dict : dictionary
        Dictionary with structure { 'same' : [ list of book pairs ( book1, book2 ) ],
                                    'different' : [ list of book pairs ( book1, book2 ) ] }
        In the 'same' list, book1 and book2 have the same author/subject/time period, while in the 'different'
        list, they are from different groups.

    alphas : list
        List of alpha parameter values.
    
    OPTIONAL:
    resample_size : integer
        Desired resample size (leave as None to use original texts).
    
    weights : boolean
        Relevant only to JSD, determines how to weight each distribution. See 'weights' parameter 
        in src/jsd.py to observe how it is used.

    metric : function
        Function for computing dissimilarity between texts.

    Returns
    -------
     : list
        The output is a list, where each value is P(X < Y) for a particular alpha.
    '''
    alpha_probs = []
    for alpha in alphas:
        new_prob = prob_x_less_than_y_freq_new(pair_dict, metric, add_parameter = alpha, 
                                    weights=weights, resample_size=resample_size)
        alpha_probs.append(new_prob)
        print(f'{alpha} : {new_prob},', sep=' ', end=' ', flush=True)
    
    return alpha_probs



