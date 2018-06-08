import numpy as np
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


class DecisionClassifier:
    def __init__(self, dmc_kwargs):
        self.random_state = np.random.RandomState(1234)
        self.train_dataset_percentage = dmc_kwargs['train_dataset_percentage']
        self.test_dataset_percentage = dmc_kwargs['test_dataset_percentage']
        self.model_type = dmc_kwargs['model_type']
        self.model_params = dmc_kwargs['dmc_model_params']
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

    def split_dataset(self, X, y):
        print("Splitting preprocessed dataset for the DecisionClassifier")
        num_steps, num_docs, num_features = X.shape
        num_training = int(np.round(num_docs * self.train_dataset_percentage))
        num_test = int(np.round(num_docs * self.test_dataset_percentage))
        if num_docs < num_training + num_test:
            print("The training-test splits must sum to one or less.")
        return X[:, 0:num_training, :], y[:, 0:num_training], X[:, num_training:, :], y[:, num_training:]

    def flatten_dataset(self, X, y):
        num_steps, num_docs, num_features = X.shape
        new_X = X.reshape((num_steps * num_docs, num_features))
        new_y = y.reshape((num_steps * num_docs))
        return new_X, new_y

    def fit(self, Xtrain, ytrain):
        print("Training PartialInformationClassifier")
        Xtrain, ytrain = self.flatten_dataset(Xtrain, ytrain)
        self.clf.fit(Xtrain, ytrain)

    def predict(self, Xtest):
        print("Predicting with DecisionClassifier")
        num_steps, num_docs, num_features = Xtest.shape
        predictions_list = []
        # We initialise the time_to_stop_reading array with the value (num_steps - 1) because if dmc never decides to
        # stop, the time to stop will be when the document is finish.
        time_to_stop_reading = (num_steps - 1) * np.ones(num_docs, dtype=int)
        for idx, step_data in enumerate(Xtest):
            predictions_step_data = self.clf.predict(step_data)
            predictions_list.append(predictions_step_data)
            for j in range(num_docs):
                if (time_to_stop_reading[j] == (num_steps - 1)) and (predictions_step_data[j] == 1):
                    time_to_stop_reading[j] = idx

        predictions = np.array(predictions_list)
        return predictions, time_to_stop_reading
