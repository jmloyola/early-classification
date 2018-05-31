import early_classification_utils as ut
import pprint as pp
import glob
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import ShuffleSplit


class PartialInformationClassifier:
    def __init__(self, cpi_kwargs, dictionary):
        print("Creando clase PartialInformationClassifier con los siguientes parámetros:")
        print(cpi_kwargs)
        self.random_state = np.random.RandomState(1234)
        self.window_size = cpi_kwargs['window_size']
        self.train_dataset_percentage = cpi_kwargs['train_dataset_percentage']
        self.test_dataset_percentage = cpi_kwargs['test_dataset_percentage']
        self.doc_rep = cpi_kwargs['doc_rep']
        self.model_type = cpi_kwargs['model_type']
        self.model_params = cpi_kwargs['cpi_model_params']
        self.dictionary = dictionary
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
        self.clf.fit(Xtrain, ytrain)


    def predict(self, Xtest):
        print("Predicting with PartialInformationClassifier")
        return None


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


def save_results(dataset, doc_representation, file_name, metadata, model_names, x, y):
    file_path = 'results/{}/{}/'.format(dataset, doc_representation, file_name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    results_file_path = file_path + file_name + '.pickle'

    with open(results_file_path, 'wb') as fp:
        pickle.dump(metadata, fp)
        pickle.dump(model_names, fp)
        pickle.dump(x, fp)
        pickle.dump(y, fp)

    return results_file_path


def build_path_to_plot(dataset, doc_representation, file_name, metadata):
    file_path = 'plots/{}/{}/'.format(dataset, doc_representation, file_name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    metadata_file_path = file_path + file_name + '.metadata'
    plot_file_path = file_path + file_name + '.pdf'

    with open(metadata_file_path, 'wb') as fp:
        pickle.dump(metadata, fp)

    return plot_file_path


def create_model(model_type, model_params):
    clf = None

    if model_type == 'DecisionTreeClassifier':
        clf = DecisionTreeClassifier(**model_params)  # **: Unpack dictionary operator.
    elif model_type == 'MultinomialNB':
        clf = MultinomialNB(**model_params)
    elif model_type == 'BernoulliNB':
        clf = BernoulliNB()  # This model doesn't have any parameters.
    elif model_type == 'GaussianNB':
        clf = GaussianNB()  # This model doesn't have any parameters.
    elif model_type == 'KNeighborsClassifier':
        clf = KNeighborsClassifier(**model_params)
    elif model_type == 'LinearSVC':
        clf = LinearSVC(**model_params)
    elif model_type == 'LogisticRegression':
        clf = LogisticRegression(**model_params)
    elif model_type == 'MLPClassifier':
        clf = MLPClassifier(**model_params)
    elif model_type == 'RandomForestClassifier':
        clf = RandomForestClassifier(**model_params)
    elif model_type == 'RidgeClassifier':
        clf = RidgeClassifier(**model_params)

    return clf


def init_metadata():
    metadata = dict()
    metadata['dataset'] = 'r8-all-terms-clean'
    metadata['doc_rep'] = 'word_tf'
    metadata['windows_size'] = 3

    metadata['models'] = []

    _model_name = 'RidgeClassifier'
    _model_params = dict()
    _model_params['solver'] = 'auto'
    _model_params['random_state'] = 0
    metadata['models'].append((_model_name, _model_params))

    _model_name = 'RandomForestClassifier'
    _model_params = dict()
    _model_params['criterion'] = 'gini'
    _model_params['random_state'] = 0
    metadata['models'].append((_model_name, _model_params))

    _model_name = 'MLPClassifier'
    _model_params = dict()
    _model_params['hidden_layer_sizes'] = 25
    _model_params['activation'] = 'relu'
    _model_params['solver'] = 'adam'
    _model_params['learning_rate'] = 'constant'
    _model_params['random_state'] = 0
    metadata['models'].append((_model_name, _model_params))

    _model_name = 'LogisticRegression'
    _model_params = dict()
    _model_params['C'] = 2
    _model_params['solver'] = 'liblinear'
    _model_params['n_jobs'] = 8
    _model_params['random_state'] = 0
    metadata['models'].append((_model_name, _model_params))

    _model_name = 'LinearSVC'
    _model_params = dict()
    _model_params['C'] = 2
    _model_params['multi_class'] = 'ovr'
    _model_params['random_state'] = 0
    metadata['models'].append((_model_name, _model_params))

    _model_name = 'KNeighborsClassifier'
    _model_params = dict()
    _model_params['n_neighbors'] = 5
    _model_params['weights'] = 'uniform'
    _model_params['algorithm'] = 'auto'
    _model_params['p'] = 2
    _model_params['n_jobs'] = 8
    metadata['models'].append((_model_name, _model_params))

    _model_name = 'GaussianNB'
    _model_params = dict()
    metadata['models'].append((_model_name, _model_params))

    _model_name = 'BernoulliNB'
    _model_params = dict()
    _model_params['alpha'] = 1.0
    _model_params['binarize'] = 0.0
    _model_params['fit_prior'] = False
    metadata['models'].append((_model_name, _model_params))

    _model_name = 'MultinomialNB'
    _model_params = dict()
    _model_params['alpha'] = 0.50
    _model_params['fit_prior'] = True
    metadata['models'].append((_model_name, _model_params))

    _model_name = 'DecisionTreeClassifier'
    _model_params = dict()
    _model_params['criterion'] = 'gini'
    _model_params['random_state'] = 0
    metadata['models'].append((_model_name, _model_params))

    return metadata


def main():
    metadata = init_metadata()
    print("Experiment's metadata:")
    pp.pprint(metadata)

    exists_preprocessed_dataset, path = preprocessed_dataset_exists(metadata['dataset'], metadata['doc_rep'])
    if exists_preprocessed_dataset:
        dictionary, X_train, y_train, unique_labels, test_data, y_test = load_preprocessed_dataset(path)
    else:
        # Search dataset path. List containing test and train corpus (in that order).
        dataset_paths = glob.glob('dataset/**/{}_*.txt'.format(metadata['dataset']), recursive=True)

        dictionary = ut.build_dict(path=dataset_paths[1], min_word_length=3,
                                   representation=metadata['doc_rep'])
        train_data = ut.transform_into_numeric_array(path=dataset_paths[1], dictionary=dictionary)

        X_train = ut.build_dataset_matrix(dataset=train_data, dictionary=dictionary,
                                          representation=metadata['doc_rep'])

        y_train, unique_labels = ut.get_labels(path=dataset_paths[1])

        test_data = ut.transform_into_numeric_array(path=dataset_paths[0], dictionary=dictionary)
        y_test, _ = ut.get_labels(path=dataset_paths[0], unique_labels=unique_labels)

        save_preprocessed_dataset(metadata['dataset'], metadata['doc_rep'], dictionary, X_train, y_train, unique_labels, test_data, y_test)

    plt.style.use('seaborn-poster')
    # seaborn-darkgrid
    # seaborn-colorblind
    # Any of this could be: ['seaborn-paper', 'seaborn-poster', 'seaborn-darkgrid', 'fast', 'ggplot', 'seaborn-pastel',
    # 'bmh', 'seaborn-deep', 'fivethirtyeight', 'grayscale', 'seaborn', 'seaborn-talk', 'seaborn-bright',
    # 'seaborn-white', 'seaborn-ticks', 'seaborn-dark-palette', 'seaborn-dark', '_classic_test', 'seaborn-colorblind',
    # 'dark_background', 'seaborn-whitegrid', 'seaborn-notebook', 'Solarize_Light2', 'seaborn-muted', 'classic']

    # To make the plot non-interactive we use plt.ioff().
    # This lets us plot at the end of the program and make block the execution until we close the plotting windows
    plt.ioff()
    # plt.ion()

    fig = plt.figure()

    docs_len = [len(t) for t in test_data]
    classifiers = []
    model_names = []
    model_ys = []
    for model_tuple in metadata['models']:
        model_type = model_tuple[0]
        model_params = model_tuple[1]
        print("Model: {} with params: {}".format(model_type, model_params))
        x = []
        y = []
        model_names.append(model_type)

        clf = create_model(model_type, model_params)

        full_model_params = clf.get_params()
        print("Model params: " + str(clf.get_params()))

        exists, path = model_exists(metadata['dataset'], metadata['doc_rep'], model_type, full_model_params)
        if exists:
            clf = load_model(path)
        else:
            clf = clf.fit(X_train, y_train)
            save_model(metadata['dataset'], metadata['doc_rep'], model_type, full_model_params, clf)

        classifiers.append(clf)

        for percentage in range(1, 101, 1):
            # We obtain the partial document.
            docs_partial_len = [int(round(l*percentage/100)) for l in docs_len]
            partial_test_data = [t[:docs_partial_len[idx]] for idx, t in enumerate(test_data)]
            X_test = ut.build_dataset_matrix(dataset=partial_test_data, dictionary=dictionary,
                                             representation=metadata['doc_rep'])
            predictions_test = clf.predict(X_test)
            mean_accuracy = np.mean(predictions_test == y_test)

            x.append(percentage)
            y.append(mean_accuracy)

        plt.plot(x, y, '-', label=model_type)
        plt.xlabel('Percentage of the document read')
        plt.ylabel('Accuracy')
        plt.title('Classification with Partial Information {}'.format(metadata['dataset']))

        # min_y = min(y)
        plt.axis([0, 100, 0, 1])
        plt.grid(True)
        plt.legend(loc='best')

        model_ys.append(y)

    fig1 = plt.gcf()  # get current figure
    plt.show()  # After the image is shown, this creates a new blank image.

    plot_file_path = build_path_to_plot(metadata['dataset'], metadata['doc_rep'], 'ModelsComparison', metadata)
    fig1.savefig(plot_file_path, dpi=3000, format='pdf')

    save_results(metadata['dataset'], metadata['doc_rep'], 'ModelsComparison', metadata, model_names, x, model_ys)


if __name__ == '__main__':
    main()
