import os
import pickle
import numpy as np
from collections import Counter

GLOBAL_BASE_DIR = os.getcwd()


def read_raw_dataset(path):
    """"
    Read raw dataset and returns a list of tuples (label,document) for every document in the dataset.

    Inputs:
    - path: path to the file of the raw dataset

    Output: List of tuples (label, document)
    """
    input_file_path = os.path.join(GLOBAL_BASE_DIR, path)

    # Store the dataset as a list of tuples (label, document).
    dataset = []
    i = 0
    with open(input_file_path, 'r') as f:
        for line in f:
            try:
                label, document = line.split(maxsplit=1)
            except:
                print(
                    'Error while reading dataset: {}. The {}º line does not follow the form \"label\tdocument\"'.format(
                        path, i))
                continue
            # Remove new line character from document.
            document = document[:-1]
            dataset.append((label, document))
            i = i + 1
    return dataset


def read_pickle_dataset(path):
    """"
    Read pickle dataset and returns a list of tuples (label,document) for every document in the dataset.

    Inputs:
    - path: path to the file of the raw dataset

    Output: List of tuples (label, document)
    """
    input_file_path = os.path.join(GLOBAL_BASE_DIR, path)
    print(input_file_path)
    with open(input_file_path, 'rb') as fp:
        dataset = pickle.load(fp)
    return dataset


def raw_to_pickle_dataset(raw_dataset, path):
    """"
    Save the raw dataset to a standard form (list of tuples (label,document)) using pickle.

    Inputs:
    - raw_dataset: list containing the raw dataset tuples (label, document)
    - path: path to the file of the raw dataset
    """
    output_file_path = os.path.join(GLOBAL_BASE_DIR, path)
    with open(output_file_path, 'wb') as fp:
        pickle.dump(raw_dataset, fp)


def build_dict(path, min_word_length=0, max_number_words=None, representation='word tf'):
    """
    Returns a dictionary with the words of the train dataset.
    The dictionary contains the words with the index sorting by number of times each word appear.
    It should be noted that the value zero of the index is reserved for the words UNKOWN.

    Inputs:
    - path: path to the file containing the raw dataset.
    - min_word_length: minimum number of characters that every word in the new dictionary must have.
    - max_number_words: maximum number of words for the dictionary.
    - representation: document representation ['word tf', 'word tf-idf', 'character 3-gram tf',
                                               'character 3-gram tf-idf', 'word 3-gram tf', 'word 3-gram tf-idf'].

    Output: dictionary containing the most relevant words in the corpus ordered by the amount of times they appear.
    """
    dataset = read_raw_dataset(path)
    documents = [x[1] for x in dataset]

    print('Building dictionary..')
    wordcount = dict()
    for ss in documents:
        words = ss.strip().lower().split()
        for w in words:
            if len(w) < min_word_length:
                continue
            if w not in wordcount:
                wordcount[w] = 1
            else:
                wordcount[w] += 1

    # Para Python 3 hace falta transformar en lista. Por ejemplo: list(wordcount.values())
    counts = list(wordcount.values())
    keys = list(wordcount.keys())

    # Hace un slicing del arreglo de counts ordenados.
    # En este caso el slicing toma todos los elementos, pero el paso es negativo, indicando que lo procesa en el orden
    # inverso.
    # numpy.argsort(counts)[::-1] es lo mismo que numpy.argsort(counts)[0:len(counts):-1]
    # Assume n is the number of elements in the dimension being sliced. Then, if i is not given it defaults to 0 for
    # k > 0 and n - 1 for k < 0 . If j is not given it defaults to n for k > 0 and -n-1 for k < 0 . If k is not given
    # it defaults to 1. Note that :: is the same as : and means select all indices along this axis.
    # https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
    sorted_idx = np.argsort(counts)[::-1]

    worddict = dict()

    for idx, ss in enumerate(sorted_idx):
        worddict[keys[ss]] = idx+1  # leave 0 (UNK)

    print(np.sum(counts), ' total words ', len(keys), ' unique words')

    return worddict


def grab_data(path, dictionary):
    """
    Given the path to a dataset, this function construct a list with the words replaced for the index given by the
    dictionary.

    Inputs:
    - path: path to the file containing the raw dataset.
    - dictionary: dictionary with the words that matter.

    Output: List of documents with the words replaced for the index given by the dictionary.
    """
    dataset = read_raw_dataset(path)
    documents = [x[1] for x in dataset]

    seqs = [None] * len(documents)
    for idx, ss in enumerate(documents):
        words = ss.strip().lower().split()
        seqs[idx] = [dictionary[w] if w in dictionary else 0 for w in words]

    return seqs


def build_dataset_matrix(dataset, dictionary=None, representation='word tf', use_unknown=True):
    """
    Build numpy.array for the dataset using the representation given in the parameter.
    Only use the words that appear in the dictionary.
    Depending on the value of use_unknown, consider the unknown words.

    Inputs:
    - dataset: dataset to process.
    - dictionary: dictionary with the words that will be consider.
    - representation: document representation ['word tf', 'word tf-idf', 'character 3-gram tf',
                                               'character 3-gram tf-idf', 'word 3-gram tf', 'word 3-gram tf-idf'].
    - use_unknown: Boolean value indicating if we should consider the unknown words.

    Output: numpy.array with dataset pre-process using the given representation.
    """
    num_docs = len(dataset)
    print("num_docs: " + str(num_docs))
    num_terms_dict = len(dictionary)
    print("num_terms_dict: " + str(num_terms_dict))
    num_features = num_terms_dict + 1 if use_unknown else num_terms_dict
    print("num_features: " + str(num_features))

    dataset_matrix = np.zeros((num_docs, num_features))

    if representation == 'word tf':
        for idx, doc in enumerate(dataset):
            counter = Counter(doc)
            for key, value in counter.items():
                dataset_matrix[idx, key] = value

    elif representation == 'word tf-idf':
        for idx, doc in enumerate(dataset):
            counter = Counter(doc)
            for key, value in counter.items():
                dataset_matrix[idx, key] = value

    return dataset_matrix


def get_labels(path, unique_labels=None):
    """"
    Read raw dataset and returns a tuple (final_labels, unique_labels).
    final_labels is a numpy.array of integers containing the label of every document.
    unique_labels is list containing every label (without repetition) ordered. Only used for the test set.

    Inputs:
    - path: path to the file of the raw dataset
    - unique_labels: list containing every label (without repetition) ordered.

    Output: tuple (final_labels, unique_labels).
    - final_labels is a numpy.array of integers containing the label of every document.
    - unique_labels is list containing every label (without repetition) ordered.
    """
    input_file_path = os.path.join(GLOBAL_BASE_DIR, path)

    labels = []
    ul = [] if unique_labels is None else unique_labels
    with open(input_file_path, 'r') as f:
        for idx, line in enumerate(f):
            try:
                label, _ = line.split(maxsplit=1)
            except RuntimeWarning:  # Arbitrary exception type
                print(
                    'Error while reading dataset: {}. The {}º line does not follow the form \"label\tdocument\"'.format(
                        path, idx))
                continue
            labels.append(label)
            if (unique_labels is None) and (label not in ul):
                ul.append(label)

        ul.sort()  # Sort list of unique_labels.
    num_documents = len(labels)
    final_labels = np.empty([num_documents], dtype=int)
    for idx, l in enumerate(labels):
        final_labels[idx] = ul.index(l)

    return final_labels, ul


if __name__ == '__main__':
    # np_dict = get_dictionary('20ng-train-stemmed', dataset_type='cachopo')
    #save_dictionary('test_dataset')
    dict = build_dict('dataset/raw/test_dataset.txt', 3)
    data = grab_data('dataset/raw/test_dataset.txt', dict)
    print(type(data))
    print(str(data[0]),len(data[0]))
    print(dict)
    print('bye...')

    '''
    input_file_path = os.path.join(GLOBAL_BASE_DIR, 'dataset/raw/cachopo')
    for corpus in os.listdir(input_file_path):
        with open(os.path.join(input_file_path, corpus), 'rb') as f:
            save_dictionary(dataset_name=corpus[:-4], dataset_type='cachopo')
    '''
