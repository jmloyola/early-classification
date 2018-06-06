from early_text_classifier import EarlyTextClassifier

# dataset_name = 'r8-all-terms-clean'
#etc_kwargs = {'dataset_name': 'r8-all-terms-clean',
#etc_kwargs = {'dataset_name': 'prueba',
etc_kwargs = {'dataset_name': 'r8-all-terms-clean',
              'initial_step': 1,
              'step_size': 33}
preprocess_kwargs = {'name': 'preprocess_kwargs',
                     'test': 3.0}
cpi_model_params = dict()
cpi_model_params['alpha'] = 1.0
cpi_model_params['binarize'] = 0.0
cpi_model_params['fit_prior'] = False
cpi_kwargs = {'window_size': 5,
              'train_dataset_percentage': 0.75,
              'test_dataset_percentage': 0.25,
              'doc_rep': 'word_tf',
              'model_type': 'BernoulliNB',
              'cpi_model_params': cpi_model_params}
context_kwargs = {'number_most_common': 3}
dmc_model_params = dict()
dmc_model_params['C'] = 2
dmc_model_params['solver'] = 'liblinear'
dmc_model_params['n_jobs'] = 1
dmc_model_params['random_state'] = 0
dmc_kwargs = {'train_dataset_percentage': 0.75,
              'test_dataset_percentage': 0.25,
              'model_type': 'LogisticRegression',
              'dmc_model_params': dmc_model_params}
performance_kwargs = {'c_tp': 1.0,
                      'c_fn': 1.0,
                      'c_fp': 1.0,
                      'penalization_type': 'Losada-Crestani',
                      'time_threshold': 95}
etc = EarlyTextClassifier(etc_kwargs, preprocess_kwargs, cpi_kwargs, context_kwargs, dmc_kwargs, performance_kwargs)
etc.print_params_information()
Xtrain, ytrain, Xtest, ytest = etc.preprocess_dataset()
print(f'Xtrain.shape: {Xtrain.shape}; ytrain.shape: {ytrain.shape}; Xtest.shape: {Xtest.shape}; ytest.shape: {ytest.shape}')
etc.fit(Xtrain, ytrain)
cpi_percentages, cpi_predictions, dmc_prediction, prediction_time = etc.predict(Xtest, ytest)
etc.score(ytest, cpi_predictions, cpi_percentages, dmc_prediction, prediction_time)
etc.save_model()
