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

    def __init__(self, context_kwargs, dictionary):
        self.number_most_common = context_kwargs['number_most_common']
        self.most_common_tokens = {}
        self.tokens_stop_words = []
        self.initial_step = context_kwargs['initial_step']
        self.step_size = context_kwargs['step_size']
        self.previous_current_doc_features = None
        self.previous_cpi_features = None
        self.read_windows = 0
        self.dictionary = dictionary

    def get_training_information(self, Xtrain, ytrain):
        print("Obtaining information from the preprocessed training data")
        for key, value in self.dictionary.items():
            if key in self.english_stop_words:
                self.tokens_stop_words.append(value)

        unique_labels = np.unique(ytrain)
        for ul in unique_labels:
            counter = Counter(Xtrain[ytrain == ul].ravel())
            # We are not interested in the number of UNKOWN, end of document and fill tokens.
            del counter[0]
            del counter[-1]
            del counter[-2]
            mc = counter.most_common(self.number_most_common)
            self.most_common_tokens[ul] = [x[0] for x in mc]

    def get_current_document_features(self, partial_Xtest):
        _, docs_len = np.where(partial_Xtest == -1)
        num_docs = len(partial_Xtest)
        num_labels = len(self.most_common_tokens)
        num_terms_feature = np.zeros((num_docs, 1))
        num_unique_terms_feature = np.zeros((num_docs, 1))
        num_stop_words_feature = np.zeros((num_docs, 1))
        num_top_words_each_class_feature = np.zeros((num_docs, num_labels))
        for idx, doc in enumerate(partial_Xtest):
            num_terms_feature[idx] = np.where(doc == -1)[0]

            unique_terms = np.unique(doc)
            index_to_delete = np.where((unique_terms == -1) | (unique_terms == -2))
            unique_terms = np.delete(unique_terms, index_to_delete)
            num_unique_terms_feature[idx] = len(unique_terms)

            num_stop_words_feature[idx] = np.sum(np.isin(doc, self.tokens_stop_words))

            for key, value in self.most_common_tokens.items():
                num_top_words_each_class_feature[idx, key] = np.sum(np.isin(doc, value))

        return np.hstack((num_terms_feature, num_unique_terms_feature, num_stop_words_feature,
                          num_top_words_each_class_feature))

    def get_cpi_features(self, cpi_predictions, current_cpi_percentage):
        num_docs = cpi_predictions.shape[1]
        percentage_feature = np.array([current_cpi_percentage] * num_docs)
        percentage_feature = percentage_feature.reshape((num_docs, 1))
        return percentage_feature

    def get_historic_features(self, current_doc_features, cpi_features):
        # TODO Implement another aggregation function.
        if self.read_windows == 0:
            self.previous_current_doc_features = current_doc_features
            self.previous_cpi_features = cpi_features
        else:
            self.previous_current_doc_features += current_doc_features
            self.previous_cpi_features += cpi_features
        self.read_windows += 1

        avg_current_doc_features = self.previous_current_doc_features / self.read_windows
        avg_cpi_features = self.previous_cpi_features / self.read_windows
        return np.hstack((avg_current_doc_features, avg_cpi_features))

    def generate_dmc_dataset(self, cpi_Xtest, cpi_ytest, cpi_predictions):
        print("Generating DecisionClassifier dataset")
        num_docs = len(cpi_Xtest)
        _, docs_len = np.where(cpi_Xtest == -1)
        dmc_X = []
        dmc_y = []
        i = 0
        for p in range(self.initial_step, 101, self.step_size):
            # We obtain the partial document.
            docs_partial_len = np.round(docs_len * p / 100).astype(int)
            max_length = np.max(docs_partial_len)
            partial_cpi_Xtest = -2*np.ones((num_docs, max_length+1), dtype=int)
            for idx, pl in enumerate(docs_partial_len):
                partial_cpi_Xtest[idx, 0:pl] = cpi_Xtest[idx, 0:pl]
                partial_cpi_Xtest[idx, pl] = -1

            current_doc_features = self.get_current_document_features(partial_cpi_Xtest)
            cpi_features = self.get_cpi_features(cpi_predictions, p)
            historic_features = self.get_historic_features(current_doc_features, cpi_features)
            partial_dmc_data = np.hstack((current_doc_features, cpi_features, historic_features))
            partial_dmc_label = (cpi_predictions[i] == cpi_ytest).astype(int)

            dmc_X.append(partial_dmc_data)
            dmc_y.append(partial_dmc_label)
            i += 1
        self.previous_current_doc_features = None
        self.previous_cpi_features = None
        self.read_windows = 0
        dmc_X = np.array(dmc_X)
        dmc_y = np.array(dmc_y)
        return dmc_X, dmc_y
