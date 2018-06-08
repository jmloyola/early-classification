import numpy as np
from scipy import sparse
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import ShuffleSplit


class PartialInformationClassifier:
    def __init__(self, cpi_kwargs, dictionary):
        self.random_state = np.random.RandomState(1234)
        self.train_dataset_percentage = cpi_kwargs['train_dataset_percentage']
        self.test_dataset_percentage = cpi_kwargs['test_dataset_percentage']
        self.doc_rep = cpi_kwargs['doc_rep']
        self.model_type = cpi_kwargs['model_type']
        self.model_params = cpi_kwargs['cpi_model_params']
        self.dictionary = dictionary
        self.initial_step = cpi_kwargs['initial_step']
        self.step_size = cpi_kwargs['step_size']
        if self.model_type == 'DecisionTreeClassifier':
            self.clf = DecisionTreeClassifier(**self.model_params)  # **: Unpack dictionary operator.
        elif self.model_type == 'MultinomialNB':
            self.clf = MultinomialNB(**self.model_params)
        elif self.model_type == 'BernoulliNB':
            self.clf = BernoulliNB()  # This model doesn't have any parameters.
        elif self.model_type == 'GaussianNB':
            self.clf = GaussianNB()  # This model doesn't have any parameters.
        elif self.model_type == 'KNeighborsClassifier':
            self.clf = KNeighborsClassifier(**self.model_params)
        elif self.model_type == 'LinearSVC':
            self.clf = LinearSVC(**self.model_params)
        elif self.model_type == 'SVC':
            self.clf = SVC(**self.model_params)
        elif self.model_type == 'LogisticRegression':
            self.clf = LogisticRegression(**self.model_params)
        elif self.model_type == 'MLPClassifier':
            self.clf = MLPClassifier(**self.model_params)
        elif self.model_type == 'RandomForestClassifier':
            self.clf = RandomForestClassifier(**self.model_params)
        elif self.model_type == 'RidgeClassifier':
            self.clf = RidgeClassifier(**self.model_params)
        else:
            self.clf = None

    def split_dataset(self, Xtrain, ytrain):
        print("Splitting preprocessed dataset for the PartialInformationClassifier")
        ss = ShuffleSplit(train_size=self.train_dataset_percentage, test_size=self.test_dataset_percentage,
                          random_state=self.random_state)
        idx_train, idx_test = next(ss.split(X=Xtrain, y=ytrain))
        cpi_Xtrain, cpi_ytrain = Xtrain[idx_train], ytrain[idx_train]
        cpi_Xtest, cpi_ytest = Xtrain[idx_test], ytrain[idx_test]
        return cpi_Xtrain, cpi_ytrain, cpi_Xtest, cpi_ytest

    def get_document_representation(self, data):
        # TODO: Implement tf-idf representation.
        num_docs = len(data)
        num_features = len(self.dictionary) + 1  # We consider de UNKOWN token.
        i = []
        j = []
        v = []
        if self.doc_rep == 'term_frec':
            for idx, row in enumerate(data):
                unique, counts = np.unique(row, return_counts=True)

                index_to_delete = np.where((unique == -1) | (unique == -2))
                unique = np.delete(unique, index_to_delete)
                counts = np.delete(counts, index_to_delete)

                i.extend([idx] * len(unique))
                j.extend(unique.tolist())
                v.extend(counts.tolist())
        sparse_matrix = sparse.coo_matrix((v, (i, j)), shape=(num_docs, num_features)).tocsr()
        return sparse_matrix

    def fit(self, Xtrain, ytrain):
        print("Training PartialInformationClassifier")
        Xtrain = self.get_document_representation(Xtrain)
        print(f'cpi_Xtrain_representation.shape: {Xtrain.shape}')
        self.clf.fit(Xtrain, ytrain)

    def predict(self, Xtest):
        print("Predicting with PartialInformationClassifier")
        num_docs = len(Xtest)
        # Remember that we used the number -1 to represent the end of the document.
        # Here we search for this token.
        # np.where gives us the index of rows and columns where the condition is True.
        # In this case, where are not interested in the rows indices.
        _, docs_len = np.where(Xtest == -1)
        percentages = []
        preds = []
        for p in range(self.initial_step, 101, self.step_size):
            # We obtain the partial document.
            docs_partial_len = np.round(docs_len * p / 100).astype(int)
            max_length = np.max(docs_partial_len)
            partial_Xtest = -2*np.ones((num_docs, max_length+1), dtype=int)
            for idx, pl in enumerate(docs_partial_len):
                partial_Xtest[idx, 0:pl] = Xtest[idx, 0:pl]
                partial_Xtest[idx, pl] = -1
            partial_Xtest = self.get_document_representation(partial_Xtest)
            if p ==self.initial_step:
                print(f'cpi_partial[i]_Xtest_representation.shape: {partial_Xtest.shape}')
            predictions_test = self.clf.predict(partial_Xtest)

            percentages.append(p)
            preds.append(predictions_test)
        percentages = np.array(percentages)
        preds = np.array(preds)
        return preds, percentages
