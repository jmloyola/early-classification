import numpy as np


class DecisionClassifier:
    def __init__(self, dmc_kwargs, verbose=True):
        self.train_dataset_percentage = dmc_kwargs['train_dataset_percentage']
        self.test_dataset_percentage = dmc_kwargs['test_dataset_percentage']
        self.clf = dmc_kwargs['dmc_clf']
        self.verbose = verbose

    def verboseprint(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def split_dataset(self, X, y):
        self.verboseprint("Splitting preprocessed dataset for the DecisionClassifier")
        num_steps, num_docs, num_features = X.shape
        num_training = int(np.round(num_docs * self.train_dataset_percentage))
        num_test = int(np.round(num_docs * self.test_dataset_percentage))
        if num_docs < num_training + num_test:
            self.verboseprint("The training-test splits must sum to one or less.")
        return X[:, 0:num_training, :], y[:, 0:num_training], X[:, num_training:, :], y[:, num_training:]

    def flatten_dataset(self, X, y):
        num_steps, num_docs, num_features = X.shape
        new_X = X.reshape((num_steps * num_docs, num_features))
        new_y = y.reshape((num_steps * num_docs))
        return new_X, new_y

    def fit(self, Xtrain, ytrain):
        self.verboseprint("Training PartialInformationClassifier")
        Xtrain, ytrain = self.flatten_dataset(Xtrain, ytrain)
        self.clf.fit(Xtrain, ytrain)

    def predict(self, Xtest):
        self.verboseprint("Predicting with DecisionClassifier")
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
