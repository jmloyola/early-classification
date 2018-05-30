import numpy as np
from collections import Counter


class ContextInformation:
    english_stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                          'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
                          'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
                          'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be',
                          'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
                          'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
                          'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
                          'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
                          'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
                          'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
                          'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just', 'don', 'should',
                          'now']

    def __init__(self, context_kwargs):
        print("Creando clase ContextInformation con los siguientes parámetros:")
        print(context_kwargs)
        self.number_most_common = context_kwargs['number_most_common']
        self.most_common_tokens = {}
        self.tokens_stop_words = []

    def get_training_information(self, Xtrain, ytrain, dictionary):
        print("Obtaining information from the preprocessed training data")
        for key, value in dictionary.items():
            if key in self.english_stop_words:
                self.tokens_stop_words.append(value)

        unique_labels = np.unique(ytrain)
        for ul in unique_labels:
            counter = Counter(Xtrain[ytrain == ul].ravel())
            # We are not interested in the number of UNKOWN and end of document tokens.
            del counter[0]
            del counter[-1]
            mc = counter.most_common(self.number_most_common)
            self.most_common_tokens[ul] = [x[0] for x in mc]

    def generate_dmc_dataset(self, cpi_Xtest, cpi_ytest, cpi_prediction, dmc_kwargs):
        print("Generating DecisionClassifier dataset")
        return None, None

    def test(self):
        print("Nothing")