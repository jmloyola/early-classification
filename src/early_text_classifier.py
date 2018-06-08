import glob
from context_information import ContextInformation
from partial_information_classifier import PartialInformationClassifier
from decision_classifier import DecisionClassifier
import numpy as np
from sklearn.metrics import recall_score, accuracy_score, f1_score, precision_score, confusion_matrix, classification_report
import pprint as pp
import pickle
import os


class EarlyTextClassifier:
    def __init__(self, etc_kwargs, preprocess_kwargs, cpi_kwargs, context_kwargs, dmc_kwargs):
        self.dataset_path = etc_kwargs['dataset_path']
        self.dataset_name = etc_kwargs['dataset_name']
        self.initial_step = etc_kwargs['initial_step']
        self.step_size = etc_kwargs['step_size']
        self.preprocess_kwargs = preprocess_kwargs
        self.cpi_kwargs = cpi_kwargs
        self.context_kwargs = context_kwargs
        self.dmc_kwargs = dmc_kwargs
        self.cpi_kwargs['initial_step'] = self.initial_step
        self.cpi_kwargs['step_size'] = self.step_size
        self.context_kwargs['initial_step'] = self.initial_step
        self.context_kwargs['step_size'] = self.step_size
        self.dictionary = None
        self.ci = None
        self.cpi = None
        self.dmc = None
        self.unique_labels = None
        self.is_loaded = False

        self.load_model()

    def has_same_parameters(self, model):
        if (self.dataset_name == model.dataset_name) and \
                (self.initial_step == model.initial_step) and \
                (self.step_size == model.step_size) and \
                (self.preprocess_kwargs == model.preprocess_kwargs) and \
                (self.cpi_kwargs == model.cpi_kwargs) and \
                (self.context_kwargs == model.context_kwargs) and \
                (self.dmc_kwargs == model.dmc_kwargs):
            return True
        else:
            return False

    def copy_attributes(self, model):
        self.ci = model.ci
        self.cpi = model.cpi
        self.dmc = model.dmc

    def load_model(self):
        possible_files = glob.glob(f'models/{self.dataset_name}/*.pickle')
        for file in possible_files:
            with open(file, 'rb') as f:
                loaded_model = pickle.load(f)
                if self.has_same_parameters(loaded_model):
                    print('Model already trained. Loading it.')
                    self.copy_attributes(loaded_model)
                    self.is_loaded = True

    def print_params_information(self):
        print("Dataset name: {}".format(self.dataset_name))
        print("Dataset path: {}".format(self.dataset_path))
        print('-'*80)
        print('Pre-process params:')
        pp.pprint(self.preprocess_kwargs)
        print('-' * 80)
        print('CPI params:')
        pp.pprint(self.cpi_kwargs)
        print('-' * 80)
        print('Context Information params:')
        pp.pprint(self.context_kwargs)
        print('-' * 80)
        print('DMC params:')
        pp.pprint(self.dmc_kwargs)
        print('-' * 80)

    def preprocess_dataset(self):
        print('Pre-processing dataset')
        # Search dataset for the dataset, both training and test sets.
        train_path = glob.glob(f'{self.dataset_path}/**/{self.dataset_name}_train.txt', recursive=True)[0]
        test_path = glob.glob(f'{self.dataset_path}/**/{self.dataset_name}_test.txt', recursive=True)[0]

        self.dictionary = build_dict(path=train_path, **self.preprocess_kwargs)

        Xtrain = transform_into_numeric_array(path=train_path, dictionary=self.dictionary)
        ytrain, self.unique_labels = get_labels(path=train_path)

        Xtest = transform_into_numeric_array(path=test_path, dictionary=self.dictionary)
        ytest, _ = get_labels(path=test_path, unique_labels=self.unique_labels)

        print(f'Xtrain.shape: {Xtrain.shape}')
        print(f'ytrain.shape: {ytrain.shape}')
        print(f'Xtest.shape: {Xtest.shape}')
        print(f'ytest.shape: {ytest.shape}')

        return Xtrain, ytrain, Xtest, ytest

    def fit(self, Xtrain, ytrain):
        if not self.is_loaded:
            print('Training EarlyTextClassifier model')
            self.ci = ContextInformation(self.context_kwargs, self.dictionary)
            self.ci.get_training_information(Xtrain, ytrain)

            self.cpi = PartialInformationClassifier(self.cpi_kwargs, self.dictionary)
            cpi_Xtrain, cpi_ytrain, cpi_Xtest, cpi_ytest = self.cpi.split_dataset(Xtrain, ytrain)

            print(f'cpi_Xtrain.shape: {cpi_Xtrain.shape}')
            print(f'cpi_ytrain.shape: {cpi_ytrain.shape}')
            print(f'cpi_Xtest.shape: {cpi_Xtest.shape}')
            print(f'cpi_ytest.shape: {cpi_ytest.shape}')

            self.cpi.fit(cpi_Xtrain, cpi_ytrain)
            cpi_predictions, cpi_percentages = self.cpi.predict(cpi_Xtest)

            dmc_X, dmc_y = self.ci.generate_dmc_dataset(cpi_Xtest, cpi_ytest, cpi_predictions)

            self.dmc = DecisionClassifier(self.dmc_kwargs)
            dmc_Xtrain, dmc_ytrain, dmc_Xtest, dmc_ytest = self.dmc.split_dataset(dmc_X, dmc_y)

            print(f'dmc_Xtrain.shape: {dmc_Xtrain.shape}')
            print(f'dmc_ytrain.shape: {dmc_ytrain.shape}')
            print(f'dmc_Xtest.shape: {dmc_Xtest.shape}')
            print(f'dmc_ytest.shape: {dmc_ytest.shape}')

            self.dmc.fit(dmc_Xtrain, dmc_ytrain)
            dmc_prediction, _ = self.dmc.predict(dmc_Xtest)
        else:
            print('EarlyTextClassifier model already trained')

    def predict(self, Xtest, ytest):
        print('Predicting with the EarlyTextClassifier model')
        cpi_predictions, cpi_percentages = self.cpi.predict(Xtest)
        dmc_X, dmc_y = self.ci.generate_dmc_dataset(Xtest, ytest, cpi_predictions)
        dmc_prediction, prediction_time = self.dmc.predict(dmc_X)
        return cpi_percentages, cpi_predictions, dmc_prediction, prediction_time, dmc_y

    def time_penalization(self, k, penalization_type, time_threshold):
        if penalization_type == 'Losada-Crestani':
            return 1.0 - ((1.0 + np.exp(k - time_threshold))**(-1))
        return 0.0

    def score(self, y_true, cpi_prediction, cpi_percentages, prediction_time, penalization_type, time_threshold, costs,
              print_ouput=True):
        y_pred = []
        k = []
        num_docs = len(y_true)
        for i in range(num_docs):
            t = prediction_time[i]
            p = cpi_percentages[prediction_time[i]]
            y_pred.append(cpi_prediction[t, i])
            k.append(p)
        y_pred = np.array(y_pred)
        k = np.array(k)

        error_score = np.zeros_like(y_true)
        if len(self.unique_labels) > 2:
            for idx in range(num_docs):
                if y_true[idx] == y_pred[idx]:
                    error_score[idx] = self.time_penalization(k[idx], penalization_type, time_threshold) * costs['c_tp']
                else:
                    error_score[idx] = costs['c_fn'] + np.sum(y_true == y_true[idx]) / num_docs
        else:
            for idx in range(num_docs):
                if (y_true[idx] == 1) and (y_pred[idx] == 1):
                    error_score[idx] = self.time_penalization(k[idx], penalization_type, time_threshold) * costs['c_tp']
                elif (y_true[idx] == 1) and (y_pred[idx] == 0):
                    error_score[idx] = costs['c_fn']
                elif (y_true[idx] == 0) and (y_pred[idx] == 1):
                    if costs['c_fp'] == 'proportion_positive_cases':
                        # TODO: document this case.
                        error_score[idx] = np.sum(y_true == 1) / num_docs
                    else:
                        error_score[idx] = costs['c_fp']
                elif (y_true[idx] == 0) and (y_pred[idx] == 0):
                    error_score[idx] = 0
        precision_etc = precision_score(y_true, y_pred, average='macro')
        recall_etc = recall_score(y_true, y_pred, average='macro')
        f1_etc = f1_score(y_true, y_pred, average='macro')
        accuracy_etc = accuracy_score(y_true, y_pred)
        erde_score = error_score.mean()
        confusion_matrix_etc = confusion_matrix(y_true, y_pred)
        if print_ouput:
            print(f'{"Score ETC":^50}')
            print('-'*50)
            print(f'{"Precision average=macro:":>25} {precision_etc:.3}')
            print(f'{"Recall average=macro:":>25} {recall_etc:.3}')
            print(f'{"F1 Measure average=macro:":>25} {f1_etc:.3}')
            print(f'{"Accuracy:":>25} {accuracy_etc:.3}')
            print(f'{"ERDE o=":>21}{time_threshold:<3}: {erde_score:.3}')
            print('-' * 50)
            print(classification_report(y_true, y_pred, target_names=self.unique_labels))
            # The reported averages are a prevalence-weighted macro-average across classes (equivalent to
            # precision_recall_fscore_support with average='weighted').
            # 'weighted': Calculate metrics for each label, and find their average, weighted by support (the number of true
            # instances for each label). This alters ‘macro’ to account for label imbalance; it can result in an F-score
            # that is not between precision and recall.
            print('-' * 50)
            print('Confusion matrix:')
            pp.pprint(confusion_matrix_etc)

        return erde_score

    def save_model(self):
        if not self.is_loaded:
            print('Saving model')
            existing_files = glob.glob(f'models/{self.dataset_name}/*.pickle')
            existing_file_names = [int(os.path.splitext(os.path.basename(x))[0]) for x in existing_files]
            max_file_name = max(existing_file_names) if existing_files != [] else 0

            file_path = f'models/{self.dataset_name}/{max_file_name + 1}.pickle'

            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb') as fp:
                pickle.dump(self, fp)


def read_raw_dataset(path):
    """"
    Read raw dataset and returns a list of tuples (label, document) for every document in the dataset.

    Inputs:
    - path: path to the file of the raw dataset

    Output: List of tuples (label, document)
    """
    # Store the dataset as a list of tuples (label, document).
    dataset = []
    i = 0
    with open(path, 'r') as f:
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


def build_dict(path, min_word_length=0, max_number_words=None, representation='word_tf'):
    """
    Returns a dictionary with the words of the train dataset.
    The dictionary contains the most relevant words with an index indicating its position in the list of number of times
    each word appears.
    It should be noted that the value zero of the index is reserved for the words UNKOWN.

    Inputs:
    - path: path to the file containing the raw dataset.
    - min_word_length: minimum number of characters that every word in the new dictionary must have.
    - max_number_words: maximum number of words for the dictionary. Use None to consider all the term in training.
    - representation: document representation ['word_tf', 'word_tf_idf', 'character_3_gram_tf',
                                               'character_3_gram_tf_idf', 'word_3_gram_tf', 'word_3_gram_tf_idf'].

    Output: dictionary containing the most relevant words in the corpus ordered by the amount of times they appear.
    """
    dataset = read_raw_dataset(path)
    documents = [x[1] for x in dataset]

    print('Building dictionary')
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

    print(np.sum(counts), ' total words ', len(keys), ' unique words')

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
        if (max_number_words is not None) and (max_number_words <= idx):
            print(f'Considering only {max_number_words} unique terms')
            break
        worddict[keys[ss]] = idx+1  # leave 0 (UNK)
    return worddict


def transform_into_numeric_array(path, dictionary):
    """
    Given the path to a dataset, this function transform a list of documents with a numpy array of shape (num_docs,
    max_length+1), where the words are replaced for the index given by the
    dictionary.

    Inputs:
    - path: path to the file containing the raw dataset.
    - dictionary: dictionary with the words that matter.

    Output: Numpy array of shape (num_docs, max_length+1) of documents with the words replaced for the index given by
    the dictionary.

    Note: The index the dictionary gives is the position of the word if we were to arrange them from more to less taking
    into account the number of occurrences of every words in the training dataset.
    The number 0 is reserved for the UNKOWN token and the number -1 is reserved to indicate the end of the document.

    """
    dataset = read_raw_dataset(path)
    num_docs = len(dataset)

    seqs = [None] * num_docs
    max_length = 0
    for idx, line in enumerate(dataset):
        document = line[1]
        words = document.strip().lower().split()
        seqs[idx] = [dictionary[w] if w in dictionary else 0 for w in words]
        length_doc = len(words)
        if max_length < length_doc:
            max_length = length_doc

    preprocess_dataset = -2*np.ones((num_docs, max_length+1), dtype=int)
    for idx in range(num_docs):
        length_doc = len(seqs[idx])
        preprocess_dataset[idx, 0:length_doc] = seqs[idx]
        preprocess_dataset[idx, length_doc] = -1

    return preprocess_dataset


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
    labels = []
    ul = [] if unique_labels is None else unique_labels
    with open(path, 'r') as f:
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
