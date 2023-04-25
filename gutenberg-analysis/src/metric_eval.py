'''The following functions are used in my evaluation of various distance metrics'''
from data_io import get_book
from jsd import jsdalpha
from itertools import combinations
import random
import pickle


# Resampling a book
def resample_book(book, size):
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


# New function for evaluating performance for vocabulary/word-frequency representations
def prob_x_less_than_y_freq_new(pair_dict, metric, add_parameter=None, resample_size=None, weights=False):
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


# New function for evaluating performance for embedding representations
def prob_x_less_than_y_emb_new(pair_dict, metric, add_parameter=None, model_name = 'all-MiniLM-L6-v2'):
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


# This function outputs a list, where each value is P(X < Y) for a particular alpha
def optimal_alpha_test_freq_new(pair_dict, alphas, resample_size = None, weights = False, metric = jsdalpha):
    alpha_probs = []
    for alpha in alphas:
        new_prob = prob_x_less_than_y_freq_new(pair_dict, metric, add_parameter = alpha, 
                                    weights=weights, resample_size=resample_size)
        alpha_probs.append(new_prob)
        print(f'{alpha} : {new_prob},', sep=' ', end=' ', flush=True)
    
    return alpha_probs



