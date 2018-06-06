import early_classification_utils as ut
import glob
from context_information import ContextInformation
from partial_information_classifier import PartialInformationClassifier
from decision_classifier import DecisionClassifier
import numpy as np
from sklearn.metrics import recall_score, accuracy_score, f1_score, precision_score, confusion_matrix
import pprint as pp


class EarlyTextClassifier:
    def __init__(self, etc_kwargs, preprocess_kwargs, cpi_kwargs, context_kwargs, dmc_kwargs, performance_kwargs):
        print("Creando clase EarlyTextClassifier con los siguientes parámetros:")
        print(etc_kwargs, preprocess_kwargs, cpi_kwargs, context_kwargs, dmc_kwargs, performance_kwargs)
        self.dataset_name = etc_kwargs['dataset_name']
        self.initial_step = etc_kwargs['initial_step']
        self.step_size = etc_kwargs['step_size']
        self.preprocess_kwargs = preprocess_kwargs
        self.cpi_kwargs = cpi_kwargs
        self.context_kwargs = context_kwargs
        self.dmc_kwargs = dmc_kwargs
        self.performance_kwargs = performance_kwargs
        self.cpi_kwargs['initial_step'] = self.initial_step
        self.cpi_kwargs['step_size'] = self.step_size
        self.context_kwargs['initial_step'] = self.initial_step
        self.context_kwargs['step_size'] = self.step_size
        self.dictionary = None
        self.ci = None
        self.cpi = None
        self.dmc = None
        self.unique_labels = None

    def print_params_information(self):
        print("Early Text Classification")
        print("Dataset name: {}".format(self.dataset_name))
        print('-'*80)
        print('Pre-process params:')
        print(self.preprocess_kwargs)
        print('-' * 80)
        print('CPI params:')
        print(self.cpi_kwargs)
        print('-' * 80)
        print('Context Information params:')
        print(self.context_kwargs)
        print('-' * 80)
        print('DMC params:')
        print(self.dmc_kwargs)
        print('-' * 80)
        print('Performance params:')
        print(self.performance_kwargs)
        print('-' * 80)
        print('-' * 80)

    def preprocess_dataset(self):
        # Search dataset for the dataset, both training and test sets.
        train_path = glob.glob('dataset/**/{}_train.txt'.format(self.dataset_name), recursive=True)[0]
        test_path = glob.glob('dataset/**/{}_test.txt'.format(self.dataset_name), recursive=True)[0]

        self.dictionary = ut.build_dict(path=train_path, min_word_length=2)

        Xtrain = ut.transform_into_numeric_array(path=train_path, dictionary=self.dictionary)
        ytrain, self.unique_labels = ut.get_labels(path=train_path)

        Xtest = ut.transform_into_numeric_array(path=test_path, dictionary=self.dictionary)
        ytest, _ = ut.get_labels(path=test_path, unique_labels=self.unique_labels)

        return Xtrain, ytrain, Xtest, ytest

    def fit(self, Xtrain, ytrain):
        self.ci = ContextInformation(self.context_kwargs, self.dictionary)
        self.ci.get_training_information(Xtrain, ytrain)

        self.cpi = PartialInformationClassifier(self.cpi_kwargs, self.dictionary)
        cpi_Xtrain, cpi_ytrain, cpi_Xtest, cpi_ytest = self.cpi.split_dataset(Xtrain, ytrain)
        self.cpi.fit(cpi_Xtrain, cpi_ytrain)
        cpi_predictions, cpi_percentages = self.cpi.predict(cpi_Xtest)

        dmc_X, dmc_y = self.ci.generate_dmc_dataset(cpi_Xtest, cpi_ytest, cpi_predictions)

        self.dmc = DecisionClassifier(self.dmc_kwargs)
        dmc_Xtrain, dmc_ytrain, dmc_Xtest, dmc_ytest = self.dmc.split_dataset(dmc_X, dmc_y)

        self.dmc.fit(dmc_Xtrain, dmc_ytrain)
        dmc_prediction, _ = self.dmc.predict(dmc_Xtest)

    def predict(self, Xtest, ytest):
        cpi_predictions, cpi_percentages = self.cpi.predict(Xtest)

        accuracy_cpi = np.sum(cpi_predictions == ytest, axis=1) / ytest.size
        print('*'*30)
        print('Accuracy of CPI for each percentage:')
        for i, percentage in enumerate(cpi_percentages):
            print(f'{percentage} % --> {accuracy_cpi[i]:.3}')
        print('*' * 30)

        dmc_X, dmc_y = self.ci.generate_dmc_dataset(Xtest, ytest, cpi_predictions)
        dmc_prediction, prediction_time = self.dmc.predict(dmc_X)

        accuracy_dmc = np.sum(dmc_prediction == dmc_y, axis=1) / dmc_y.size
        print('*' * 30)
        print('Accuracy of DMC for each percentage:')
        for i, percentage in enumerate(cpi_percentages):
            print(f'{percentage} % --> {accuracy_dmc[i]:.3}')
        print('*' * 30)

        return cpi_percentages, cpi_predictions, dmc_prediction, prediction_time

    def get_time_penalization(self, k):
        if self.performance_kwargs['penalization_type'] == 'Losada-Crestani':
            return 1.0 - ((1.0 + np.exp(k - self.performance_kwargs['time_threshold']))**(-1))
        return 0.0

    def score(self, y_true, cpi_prediction, cpi_percentages, dmc_prediction, prediction_time):
        # TODO
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
                    error_score[idx] = self.get_time_penalization(k[idx]) * self.performance_kwargs['c_tp']
                else:
                    error_score[idx] = self.performance_kwargs['c_fn'] + np.sum(y_true == y_true[idx]) / num_docs
        else:
            for idx in range(num_docs):
                if (y_true[idx] == 1) and (y_pred[idx] == 1):
                    error_score[idx] = self.get_time_penalization(k[idx]) * self.performance_kwargs['c_tp']
                elif (y_true[idx] == 1) and (y_pred[idx] == 0):
                    error_score[idx] = self.performance_kwargs['c_fn']
                elif (y_true[idx] == 0) and (y_pred[idx] == 1):
                    # error_score[idx] = self.performance_kwargs['c_fp']
                    error_score[idx] = np.sum(y_true == 1) / num_docs
                elif (y_true[idx] == 0) and (y_pred[idx] == 0):
                    error_score[idx] = 0
        erde_score = error_score.mean()
        precision_etc = precision_score(y_true, y_pred, average='micro')
        recall_etc = recall_score(y_true, y_pred, average='micro')
        f1_etc = f1_score(y_true, y_pred, average='micro')
        accuracy_etc = accuracy_score(y_true, y_pred)
        #confusion_matrix_etc = confusion_matrix(y_true, y_pred, self.unique_labels)
        confusion_matrix_etc = confusion_matrix(y_true, y_pred)
        print('*' * 30)
        print('Score ETC:')
        print(' - '*10)
        print(f'Precision: {precision_etc:.3}')
        print(f'Recall: {recall_etc:.3}')
        print(f'F1 Measure: {f1_etc:.3}')
        print(f'Accuracy: {accuracy_etc:.3}')
        print(f'ERDE o={self.performance_kwargs["time_threshold"]}: {erde_score:.3}')
        print('Confusion matrix:')
        pp.pprint(confusion_matrix_etc)
        print('*' * 30)
        return
