import early_classification_utils as ut
import glob
from context_information import ContextInformation
from partial_information_classifier import PartialInformationClassifier
from decision_classifier import DecisionClassifier


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
        ytrain, unique_labels = ut.get_labels(path=train_path)

        Xtest = ut.transform_into_numeric_array(path=test_path, dictionary=self.dictionary)
        ytest, _ = ut.get_labels(path=test_path, unique_labels=unique_labels)

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
        dmc_X, dmc_y = self.ci.generate_dmc_dataset(Xtest, ytest, cpi_predictions)
        dmc_prediction, prediction_time = self.dmc.predict(dmc_X)
        return cpi_percentages, cpi_predictions, dmc_prediction, prediction_time

    def score(self, ytest, cpi_prediction, dmc_prediction, prediction_time):
        return
