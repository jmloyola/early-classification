import early_classification_utils as ut
import pprint as pp
import glob
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from sklearn import tree


def preprocessed_dataset_exists(dataset, doc_representation):
    path = glob.glob(
        'preprocessed_dataset/{}/{}/dict_Xtrain_ytrain_testData_ytest.pickle'.format(dataset, doc_representation))
    if path:
        return True, path[0]  # We should return the only element in the list.
    return False, None


def load_preprocessed_dataset(path_preprocessed_dataset):
    with open(path_preprocessed_dataset, 'rb') as fp:
        dictionary = pickle.load(fp)
        X_train = pickle.load(fp)
        y_train = pickle.load(fp)
        unique_labels = pickle.load(fp)
        test_data = pickle.load(fp)
        y_test = pickle.load(fp)
    return dictionary, X_train, y_train, unique_labels, test_data, y_test


def save_preprocessed_dataset(dataset, doc_representation, dictionary, X_train, y_train, unique_labels, test_data, y_test):
    file_path = 'preprocessed_dataset/{}/{}/dict_Xtrain_ytrain_testData_ytest.pickle'.format(dataset, doc_representation)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as fp:
        pickle.dump(dictionary, fp)
        pickle.dump(X_train, fp)
        pickle.dump(y_train, fp)
        pickle.dump(unique_labels, fp)
        pickle.dump(test_data, fp)
        pickle.dump(y_test, fp)
    return file_path


def model_exists(dataset, doc_representation, model_type, model_parameters):
    possible_files = glob.glob('models/{}/{}/{}/*.pickle'.format(dataset, doc_representation, model_type))
    for file in possible_files:
        with open(file, 'rb') as f:
            loaded_parameters = pickle.load(f)
            if loaded_parameters == model_parameters:
                return True, file
    return False, None


def load_model(path_model):
    with open(path_model, 'rb') as fp:
        _ = pickle.load(fp)  # First we load the model parameters, which we don't care about.
        model = pickle.load(fp)
    return model


def save_model(dataset, doc_representation, model_type, model_parameters, model):
    existing_files = glob.glob('models/{}/{}/{}/*.pickle'.format(dataset, doc_representation, model_type))
    existing_file_names = [int(os.path.splitext(os.path.basename(x))[0]) for x in existing_files]
    max_file_name = max(existing_file_names) if existing_files != [] else 0

    file_path = 'models/{}/{}/{}/{}.pickle'.format(dataset, doc_representation, model_type, max_file_name + 1)

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as fp:
        pickle.dump(model_parameters, fp)
        pickle.dump(model, fp)

    return file_path


def main():
    metadata = dict()
    metadata['dataset'] = 'r8-all-terms-clean'
    metadata['doc_representation'] = 'word_tf'
    metadata['windows_size'] = 3
    metadata['model'] = 'DecisionTreeClassifier'

    classifier_params = dict()
    classifier_params['criterion'] = 'entropy'
    classifier_params['random_state'] = 0
    metadata['classifier_params'] = classifier_params

    print("Experiment's metadata:")
    pp.pprint(metadata)

    exists_preprocessed_dataset, path = preprocessed_dataset_exists(metadata['dataset'], metadata['doc_representation'])
    if exists_preprocessed_dataset:
        dictionary, X_train, y_train, unique_labels, test_data, y_test = load_preprocessed_dataset(path)
    else:
        # Search dataset path. List containing test and train corpus (in that order).
        dataset_paths = glob.glob('dataset/**/{}_*.txt'.format(metadata['dataset']), recursive=True)

        dictionary = ut.build_dict(path=dataset_paths[1], min_word_length=3,
                                   representation=metadata['doc_representation'])
        train_data = ut.grab_data(path=dataset_paths[1], dictionary=dictionary)

        X_train = ut.build_dataset_matrix(dataset=train_data, dictionary=dictionary,
                                          representation=metadata['doc_representation'])

        y_train, unique_labels = ut.get_labels(path=dataset_paths[1])

        test_data = ut.grab_data(path=dataset_paths[0], dictionary=dictionary)
        y_test, _ = ut.get_labels(path=dataset_paths[0], unique_labels=unique_labels)

        save_preprocessed_dataset(metadata['dataset'], metadata['doc_representation'], dictionary, X_train, y_train, unique_labels, test_data, y_test)

    clf = tree.DecisionTreeClassifier(**metadata['classifier_params'])  # **: Unpack dictionary operator.
    full_model_params = clf.get_params()
    print("Model params: " + str(clf.get_params()))

    exists, path = model_exists(metadata['dataset'], metadata['doc_representation'], metadata['model'], full_model_params)
    if exists:
        clf = load_model(path)
    else:
        clf = clf.fit(X_train, y_train)
        save_model(metadata['dataset'], metadata['doc_representation'], metadata['model'], full_model_params, clf)

    max_len_test_data = np.max([len(t) for t in test_data])
    x = []
    y = []

    plt.style.use('seaborn-darkgrid')
    # seaborn-darkgrid
    # seaborn-colorblind
    # Any of this could be: ['seaborn-paper', 'seaborn-poster', 'seaborn-darkgrid', 'fast', 'ggplot', 'seaborn-pastel',
    # 'bmh', 'seaborn-deep', 'fivethirtyeight', 'grayscale', 'seaborn', 'seaborn-talk', 'seaborn-bright',
    # 'seaborn-white', 'seaborn-ticks', 'seaborn-dark-palette', 'seaborn-dark', '_classic_test', 'seaborn-colorblind',
    # 'dark_background', 'seaborn-whitegrid', 'seaborn-notebook', 'Solarize_Light2', 'seaborn-muted', 'classic']

    # To make the plot non-interactive we use plt.ioff().
    # This lets us plot at the end of the program and make block the execution until we close the plotting windows
    plt.ioff()
    fig = plt.figure()
    for idx in range(metadata['windows_size'], max_len_test_data, metadata['windows_size']):
        # We obtain the partial document.
        partial_test_data = [t[:idx] for t in test_data]
        X_test = ut.build_dataset_matrix(dataset=partial_test_data, dictionary=dictionary,
                                         representation=metadata['doc_representation'])
        predictions_test = clf.predict(X_test)
        mean_accuracy = np.mean(predictions_test == y_test)

        x.append(idx)
        y.append(mean_accuracy)

    plt.plot(x, y, '-')
    plt.show()
    # plt.savefig(fname='', dpi=3000, format='pdf')

    x = []
    y = []
    docs_len = [len(t) for t in test_data]
    fig = plt.figure()
    for percentage in range(5, 100, 5):
        # We obtain the partial document.
        docs_partial_len = [int(round(l*percentage/100)) for l in docs_len]
        partial_test_data = [t[:docs_partial_len[idx]] for idx, t in enumerate(test_data)]
        X_test = ut.build_dataset_matrix(dataset=partial_test_data, dictionary=dictionary,
                                         representation=metadata['doc_representation'])
        predictions_test = clf.predict(X_test)
        mean_accuracy = np.mean(predictions_test == y_test)

        x.append(percentage)
        y.append(mean_accuracy)

    plt.plot(x, y, '-')
    plt.show()


if __name__ == '__main__':
    main()
