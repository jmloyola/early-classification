import early_classification_utils as ut
import pprint as pp
import glob
from sklearn import tree
import numpy as np

if __name__ == '__main__':
    metadata = dict()
    metadata['dataset'] = 'r8-all-terms-clean'
    metadata['doc_representation'] = 'word tf'
    metadata['windows_size'] = 3

    classifier_info = dict()
    classifier_info['model'] = 'Naive Bayes'
    classifier_info['parameters'] = []
    metadata['classifier_info'] = classifier_info

    print("Experiment's metadata:")
    pp.pprint(metadata)

    # Search dataset path. List containing test and train corpus (in that order).
    dataset_paths = glob.glob('dataset/**/{}_*.txt'.format(metadata['dataset']), recursive=True)

    #train_dataset = ut.read_raw_dataset(dataset_paths[1])
    dictionary = ut.build_dict(path=dataset_paths[1], min_word_length=3, representation=metadata['doc_representation'])
    train_data = ut.grab_data(path=dataset_paths[1], dictionary=dictionary)
    #print(train_dataset[0][1])
    print(type(train_data))
    print(train_data[0])
    #print(train_dataset[0][1].split()[data[0].index(1)])

    X_train = ut.build_dataset_matrix(dataset=train_data, dictionary=dictionary,
                                      representation=metadata['doc_representation'])
    print(X_train[0, :])

    y_train, unique_labels = ut.get_labels(path=dataset_paths[1])
    print(y_train.shape)
    print(y_train[14])
    print(unique_labels)

    print('training model...')
    clf = tree.DecisionTreeClassifier(criterion='entropy', random_state=0)
    clf = clf.fit(X_train, y_train)
    #result = cross_val_score(clf, X_train, y_train, cv=10)
    #print(result)
    print('training model...[DONE]')

    print(str(clf.get_params()))

    test_data = ut.grab_data(path=dataset_paths[0], dictionary=dictionary)
    X_test = ut.build_dataset_matrix(dataset=test_data, dictionary=dictionary,
                                     representation=metadata['doc_representation'])
    y_test, _ = ut.get_labels(path=dataset_paths[0], unique_labels=unique_labels)

    predictions_test = clf.predict(X_test)
    score = clf.score(X_test, y_test)

    print("Predicted result:   " + str(predictions_test[0:5]))
    print("Ground-True result: " + str(y_test[0:5]))
    print("Mean accuracy:      " + str(np.mean(predictions_test[0:5] == y_test[0:5])))

    print("Final score: " + str(score))
    print("Final score: " + str(np.mean(predictions_test == y_test)))

    max_len_test_data = np.max([len(x) for x in test_data])
    for idx in range(metadata['windows_size'], max_len_test_data, metadata['windows_size']):
        # We obtain the partial document.
        partial_test_data = [x[:idx] for x in test_data]
        X_test = ut.build_dataset_matrix(dataset=partial_test_data, dictionary=dictionary,
                                         representation=metadata['doc_representation'])
        predictions_test = clf.predict(X_test)
        print()
