import early_classification_utils as ut
import glob
import numpy as np
import pickle
import os
from context_information import ContextInformation
from partial_information_classifier import PartialInformationClassifier
from decision_classifier import DecisionClassifier
import pprint as pp


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


def preprocess_dataset(dataset_name):
    # Search dataset for the dataset, both training and test sets.
    train_path = glob.glob('dataset/**/{}_train.txt'.format(dataset_name), recursive=True)[0]
    test_path = glob.glob('dataset/**/{}_test.txt'.format(dataset_name), recursive=True)[0]

    dictionary = ut.build_dict(path=train_path, min_word_length=2)

    Xtrain = ut.transform_into_numeric_array(path=train_path, dictionary=dictionary)
    ytrain, unique_labels = ut.get_labels(path=train_path)

    Xtest = ut.transform_into_numeric_array(path=test_path, dictionary=dictionary)
    ytest, _ = ut.get_labels(path=test_path, unique_labels=unique_labels)

    return Xtrain, ytrain, Xtest, ytest, dictionary


def fit(Xtrain, ytrain, cpi_kwargs, context_kwargs, dmc_kwargs, dictionary):
    ci = ContextInformation(context_kwargs)
    ci.get_training_information(Xtrain, ytrain, dictionary)

    cpi = PartialInformationClassifier(cpi_kwargs)
    cpi_Xtrain, cpi_ytrain, cpi_Xtest, cpi_ytest = cpi.split_dataset(Xtrain, ytrain)
    cpi.fit(cpi_Xtrain, cpi_ytrain)
    cpi_prediction = cpi.predict(cpi_Xtest)

    dmc_X, dmc_y = ci.generate_dmc_dataset(cpi_Xtest, cpi_ytest, cpi_prediction, dmc_kwargs)

    dmc = DecisionClassifier(dmc_kwargs)
    dmc_Xtrain, dmc_ytrain, dmc_Xtest, dmc_ytest = dmc.split_dataset(dmc_X, dmc_y)
    dmc.fit(dmc_Xtrain, dmc_ytrain)
    dmc_prediction, _ = dmc.predict(dmc_Xtest)
    return ci, cpi, dmc


def predict(Xtest, ytest, ci, cpi, dmc):
    cpi_prediction = cpi.predict(Xtest)
    dmc_X, dmc_y = ci.generate_dmc_dataset(Xtest, ytest, cpi_prediction, dmc_kwargs)
    dmc_prediction, prediction_time = dmc.predict(dmc_X)
    return cpi_prediction, dmc_prediction, prediction_time


def score(ytest, cpi_prediction, dmc_prediction, prediction_time, performance_kwargs):
    return


def main(dataset_name, preprocess_kwargs, cpi_kwargs, context_kwargs, dmc_kwargs, performance_kwargs):
    print_params_information(dataset_name, preprocess_kwargs, cpi_kwargs, context_kwargs, dmc_kwargs,
                             performance_kwargs)
    Xtrain, ytrain, Xtest, ytest, dictionary = preprocess_dataset(dataset_name)

    ci, cpi, dmc = fit(Xtrain, ytrain, cpi_kwargs, context_kwargs, dmc_kwargs, dictionary)
    cpi_prediction, dmc_prediction, prediction_time = predict(Xtest, ytest, ci, cpi, dmc)

    score(ytest, cpi_prediction, dmc_prediction, prediction_time, performance_kwargs)


if __name__ == '__main__':
    dataset_name = 'prueba'
    preprocess_kwargs = {'name': 'preprocess_kwargs', 'test': 3.0}
    cpi_kwargs = {'window_size': 5, 'train_dataset_percentage': 0.75, 'test_dataset_percentage': 0.25, 'name': 'cpi_kwargs', 'test': 3.0}
    context_kwargs = {'number_most_common': 3, 'name': 'context_kwargs', 'test': 3.0}
    dmc_kwargs = {'name': 'dmc_kwargs', 'test': 3.0}
    performance_kwargs = {'name': 'performance_kwargs', 'test': 3.0}
    main(dataset_name, preprocess_kwargs, cpi_kwargs, context_kwargs, dmc_kwargs, performance_kwargs)
