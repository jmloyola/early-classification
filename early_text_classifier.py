import early_classification_utils as ut
import glob
import numpy as np
import pickle
import os
from context_information import ContextInformation
from partial_information_classifier import PartialInformationClassifier
from decision_classifier import DecisionClassifier


def preprocess_dataset(dataset_name):
    print("Preprocessing dataset {}".format(dataset_name))
    return None


def split_dataset(preprocessed_dataset):
    print("Splitting preprocessed dataset")
    return None, None, None, None


def print_params_information(dataset_name, preprocess_kwargs, cpi_kwargs, context_kwargs, dmc_kwargs,
                             performance_kwargs):
    print("Early Text Classification")
    print("Dataset name: {}".format(dataset_name))
    print('-'*80)
    print('Pre-process params:')
    print(preprocess_kwargs)
    print('-' * 80)
    print('CPI params:')
    print(cpi_kwargs)
    print('-' * 80)
    print('Context Information params:')
    print(context_kwargs)
    print('-' * 80)
    print('DMC params:')
    print(dmc_kwargs)
    print('-' * 80)
    print('Performance params:')
    print(performance_kwargs)
    print('-' * 80)
    print('-' * 80)


def fit(Xtrain, ytrain, cpi_kwargs, context_kwargs, dmc_kwargs):
    ci = ContextInformation(context_kwargs)
    training_data_information = ci.get_training_information(Xtrain, ytrain)

    cpi = PartialInformationClassifier(cpi_kwargs)
    cpi_Xtrain, cpi_ytrain, cpi_Xtest, cpi_ytest = cpi.split_dataset(Xtrain, ytrain)
    cpi.fit(cpi_Xtrain, cpi_ytrain)
    cpi_prediction = cpi.predict(cpi_Xtest)

    dmc_X, dmc_y = ci.generate_dmc_dataset(cpi_Xtest, cpi_ytest, cpi_prediction, training_data_information)

    dmc = DecisionClassifier(dmc_kwargs)
    dmc_Xtrain, dmc_ytrain, dmc_Xtest, dmc_ytest = dmc.split_dataset(dmc_X, dmc_y)
    dmc.fit(dmc_Xtrain, dmc_ytrain)
    dmc_prediction = dmc.predict(dmc_Xtest)
    return ci, cpi, dmc


def predict(Xtest, ytest, ci, cpi, dmc, performance_kwargs):
    # TODO: FINISH!
    return


def main(dataset_name, preprocess_kwargs, cpi_kwargs, context_kwargs, dmc_kwargs, performance_kwargs):
    print_params_information(dataset_name, preprocess_kwargs, cpi_kwargs, context_kwargs, dmc_kwargs,
                             performance_kwargs)
    data = preprocess_dataset(dataset_name)
    Xtrain, ytrain, Xtest, ytest = split_dataset(data)

    ci, cpi, dmc = fit(Xtrain, ytrain, cpi_kwargs, context_kwargs, dmc_kwargs)
    predict(Xtest, ytest, ci, cpi, dmc, performance_kwargs)


if __name__ == '__main__':
    dataset_name = 'r8-all-terms-clean'
    preprocess_kwargs = {'name':'preprocess_kwargs', 'test': 3.0}
    cpi_kwargs = {'name':'cpi_kwargs', 'test': 3.0}
    context_kwargs = {'name':'context_kwargs', 'test': 3.0}
    dmc_kwargs = {'name':'dmc_kwargs', 'test': 3.0}
    performance_kwargs = {'name':'performance_kwargs', 'test': 3.0}
    main(dataset_name, preprocess_kwargs, cpi_kwargs, context_kwargs, dmc_kwargs, performance_kwargs)