import glob
from preprocess_dataset import PreprocessDataset
from context_information import ContextInformation
from partial_information_classifier import PartialInformationClassifier
from decision_classifier import DecisionClassifier
import numpy as np
from sklearn.metrics import recall_score, accuracy_score, f1_score, precision_score, confusion_matrix,\
    classification_report
import pprint as pp
import pickle
import os


class EarlyTextClassifier:
    def __init__(self, etc_kwargs, preprocess_kwargs, cpi_kwargs, context_kwargs, dmc_kwargs, unique_labels=None,
                 dictionary=None, verbose=True):
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
        self.dictionary = dictionary
        self.ci = None
        self.cpi = None
        self.dmc = None
        self.unique_labels = unique_labels
        self.is_loaded = False
        self.verbose = verbose

        self.load_model()

    def verboseprint(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def has_same_parameters(self, model):
        if (self.dataset_name == model.dataset_name) and \
                (self.initial_step == model.initial_step) and \
                (self.step_size == model.step_size) and \
                (self.preprocess_kwargs == model.preprocess_kwargs) and \
                (self.cpi_kwargs['train_dataset_percentage'] == model.cpi_kwargs['train_dataset_percentage']) and \
                (self.cpi_kwargs['test_dataset_percentage'] == model.cpi_kwargs['test_dataset_percentage']) and \
                (self.cpi_kwargs['doc_rep'] == model.cpi_kwargs['doc_rep']) and \
                (self.cpi_kwargs['cpi_clf'].get_params() == model.cpi_kwargs['cpi_clf'].get_params()) and \
                (self.context_kwargs == model.context_kwargs) and \
                (self.dmc_kwargs['train_dataset_percentage'] == model.dmc_kwargs['train_dataset_percentage']) and \
                (self.dmc_kwargs['test_dataset_percentage'] == model.dmc_kwargs['test_dataset_percentage']) and \
                (self.dmc_kwargs['dmc_clf'].get_params() == model.dmc_kwargs['dmc_clf'].get_params()):
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
                    self.verboseprint('Model already trained. Loading it.')
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
        self.verboseprint('Pre-processing dataset')
        # Search dataset for the dataset, both training and test sets.
        train_path = glob.glob(f'{self.dataset_path}/**/{self.dataset_name}_train.txt', recursive=True)[0]
        test_path = glob.glob(f'{self.dataset_path}/**/{self.dataset_name}_test.txt', recursive=True)[0]

        prep_data = PreprocessDataset(self.preprocess_kwargs, self.verbose)
        self.dictionary = prep_data.build_dict(train_path)

        Xtrain = prep_data.transform_into_numeric_array(train_path)
        ytrain, self.unique_labels = prep_data.get_labels(train_path)

        Xtest = prep_data.transform_into_numeric_array(test_path)
        ytest, _ = prep_data.get_labels(test_path)

        self.verboseprint(f'Xtrain.shape: {Xtrain.shape}')
        self.verboseprint(f'ytrain.shape: {ytrain.shape}')
        self.verboseprint(f'Xtest.shape: {Xtest.shape}')
        self.verboseprint(f'ytest.shape: {ytest.shape}')

        return Xtrain, ytrain, Xtest, ytest

    def fit(self, Xtrain, ytrain):
        if not self.is_loaded:
            self.verboseprint('Training EarlyTextClassifier model')
            self.ci = ContextInformation(self.context_kwargs, self.dictionary, self.verbose)
            self.ci.get_training_information(Xtrain, ytrain)

            self.cpi = PartialInformationClassifier(self.cpi_kwargs, self.dictionary, self.verbose)
            cpi_Xtrain, cpi_ytrain, cpi_Xtest, cpi_ytest = self.cpi.split_dataset(Xtrain, ytrain)

            self.verboseprint(f'cpi_Xtrain.shape: {cpi_Xtrain.shape}')
            self.verboseprint(f'cpi_ytrain.shape: {cpi_ytrain.shape}')
            self.verboseprint(f'cpi_Xtest.shape: {cpi_Xtest.shape}')
            self.verboseprint(f'cpi_ytest.shape: {cpi_ytest.shape}')

            self.cpi.fit(cpi_Xtrain, cpi_ytrain)
            cpi_predictions, cpi_percentages = self.cpi.predict(cpi_Xtest)

            dmc_X, dmc_y = self.ci.generate_dmc_dataset(cpi_Xtest, cpi_ytest, cpi_predictions)

            self.dmc = DecisionClassifier(self.dmc_kwargs, self.verbose)
            dmc_Xtrain, dmc_ytrain, dmc_Xtest, dmc_ytest = self.dmc.split_dataset(dmc_X, dmc_y)

            self.verboseprint(f'dmc_Xtrain.shape: {dmc_Xtrain.shape}')
            self.verboseprint(f'dmc_ytrain.shape: {dmc_ytrain.shape}')
            self.verboseprint(f'dmc_Xtest.shape: {dmc_Xtest.shape}')
            self.verboseprint(f'dmc_ytest.shape: {dmc_ytest.shape}')

            self.dmc.fit(dmc_Xtrain, dmc_ytrain)
            dmc_prediction, _ = self.dmc.predict(dmc_Xtest)
        else:
            self.verboseprint('EarlyTextClassifier model already trained')

    def predict(self, Xtest, ytest):
        self.verboseprint('Predicting with the EarlyTextClassifier model')
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

        error_score = np.zeros(num_docs)
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
            # 'weighted': Calculate metrics for each label, and find their average, weighted by support (the number of
            # true instances for each label). This alters ‘macro’ to account for label imbalance; it can result in an
            # F-score that is not between precision and recall.
            print('-' * 50)
            print('Confusion matrix:')
            pp.pprint(confusion_matrix_etc)

        return precision_etc, recall_etc, f1_etc, accuracy_etc, erde_score

    def save_model(self):
        if not self.is_loaded:
            self.verboseprint('Saving model')
            existing_files = glob.glob(f'models/{self.dataset_name}/*.pickle')
            existing_file_names = [int(os.path.splitext(os.path.basename(x))[0]) for x in existing_files]
            max_file_name = max(existing_file_names) if existing_files != [] else 0

            file_path = f'models/{self.dataset_name}/{max_file_name + 1}.pickle'

            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb') as fp:
                pickle.dump(self, fp)
        else:
            self.verboseprint('Model already in disk')
