from early_text_classifier import EarlyTextClassifier

# dataset_name = 'r8-all-terms-clean'
#etc_kwargs = {'dataset_name': 'r8-all-terms-clean',
#etc_kwargs = {'dataset_name': 'prueba',
etc_kwargs = {'dataset_name': 'prueba',
              'initial_step': 50,
              'step_size': 49}
preprocess_kwargs = {'name': 'preprocess_kwargs',
                     'test': 3.0}
cpi_model_params = dict()
cpi_model_params['C'] = 2
cpi_model_params['multi_class'] = 'ovr'
cpi_model_params['random_state'] = 0
cpi_kwargs = {'window_size': 5,
              'train_dataset_percentage': 0.75,
              'test_dataset_percentage': 0.25,
              'doc_rep': 'word_tf',
              'model_type': 'LinearSVC',
              'cpi_model_params': cpi_model_params}
context_kwargs = {'number_most_common': 3}
dmc_kwargs = {'train_dataset_percentage': 0.75,
              'test_dataset_percentage': 0.25,
              'model_type': 'LinearSVC',
              'dmc_model_params': cpi_model_params}
performance_kwargs = {'name': 'performance_kwargs',
                      'test': 3.0}
etc = EarlyTextClassifier(etc_kwargs, preprocess_kwargs, cpi_kwargs, context_kwargs, dmc_kwargs, performance_kwargs)
etc.print_params_information()
Xtrain, ytrain, Xtest, ytest = etc.preprocess_dataset()
etc.fit(Xtrain, ytrain)
cpi_percentages, cpi_predictions, dmc_prediction, prediction_time = etc.predict(Xtest, ytest)
etc.score(ytest, cpi_predictions, dmc_prediction, prediction_time)
