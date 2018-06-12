import numpy as np
from scipy import sparse
from sklearn.model_selection import ShuffleSplit


class PartialInformationClassifier:
    def __init__(self, cpi_kwargs, dictionary, verbose=True):
        self.random_state = np.random.RandomState(1234)
        self.train_dataset_percentage = cpi_kwargs['train_dataset_percentage']
        self.test_dataset_percentage = cpi_kwargs['test_dataset_percentage']
        self.doc_rep = cpi_kwargs['doc_rep']
        self.dictionary = dictionary
        self.initial_step = cpi_kwargs['initial_step']
        self.step_size = cpi_kwargs['step_size']
        self.clf = cpi_kwargs['cpi_clf']
        self.verbose = verbose

    def verboseprint(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def split_dataset(self, Xtrain, ytrain):
        self.verboseprint("Splitting preprocessed dataset for the PartialInformationClassifier")
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
        self.verboseprint("Training PartialInformationClassifier")
        Xtrain = self.get_document_representation(Xtrain)
        self.verboseprint(f'cpi_Xtrain_representation.shape: {Xtrain.shape}')
        self.clf.fit(Xtrain, ytrain)

    def predict(self, Xtest):
        self.verboseprint("Predicting with PartialInformationClassifier")
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
                self.verboseprint(f'cpi_partial[i]_Xtest_representation.shape: {partial_Xtest.shape}')
            predictions_test = self.clf.predict(partial_Xtest)

            percentages.append(p)
            preds.append(predictions_test)
        percentages = np.array(percentages)
        preds = np.array(preds)
        return preds, percentages
