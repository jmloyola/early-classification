import numpy as np

class PreprocessDataset:
    def __init__(self, preprocess_kwargs, verbose=True):
        self.min_word_length = preprocess_kwargs['min_word_length']
        self.max_number_words = preprocess_kwargs['max_number_words']
        self.representation = 'word_tf'
        self.dictionary = None
        self.unique_labels = None
        self.verbose = verbose

    def verboseprint(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def read_raw_dataset(self, path):
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

    def build_dict(self, path):
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
        dataset = self.read_raw_dataset(path)
        documents = [x[1] for x in dataset]

        self.verboseprint('Building dictionary')
        wordcount = dict()
        for ss in documents:
            words = ss.strip().lower().split()
            for w in words:
                if len(w) < self.min_word_length:
                    continue
                if w not in wordcount:
                    wordcount[w] = 1
                else:
                    wordcount[w] += 1

        # Para Python 3 hace falta transformar en lista. Por ejemplo: list(wordcount.values())
        counts = list(wordcount.values())
        keys = list(wordcount.keys())

        self.verboseprint(np.sum(counts), ' total words ', len(keys), ' unique words')

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
            if (self.max_number_words is not None) and (self.max_number_words <= idx):
                self.verboseprint(f'Considering only {self.max_number_words} unique terms')
                break
            worddict[keys[ss]] = idx + 1  # leave 0 (UNK)
        self.dictionary = worddict
        return worddict

    def transform_into_numeric_array(self, path):
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
        dataset = self.read_raw_dataset(path)
        num_docs = len(dataset)

        seqs = [None] * num_docs
        max_length = 0
        for idx, line in enumerate(dataset):
            document = line[1]
            words = document.strip().lower().split()
            seqs[idx] = [self.dictionary[w] if w in self.dictionary else 0 for w in words]
            length_doc = len(words)
            if max_length < length_doc:
                max_length = length_doc

        preprocess_dataset = -2 * np.ones((num_docs, max_length + 1), dtype=int)
        for idx in range(num_docs):
            length_doc = len(seqs[idx])
            preprocess_dataset[idx, 0:length_doc] = seqs[idx]
            preprocess_dataset[idx, length_doc] = -1

        return preprocess_dataset

    def get_labels(self, path):
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
        ul = [] if self.unique_labels is None else self.unique_labels
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
                if (self.unique_labels is None) and (label not in ul):
                    ul.append(label)

            ul.sort()  # Sort list of unique_labels.
        num_documents = len(labels)
        final_labels = np.empty([num_documents], dtype=int)
        for idx, l in enumerate(labels):
            final_labels[idx] = ul.index(l)

        self.unique_labels = ul
        return final_labels, ul
