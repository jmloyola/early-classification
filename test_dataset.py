from early_text_classifier import EarlyTextClassifier

# dataset_name = 'r8-all-terms-clean'
#etc_kwargs = {'dataset_name': 'r8-all-terms-clean',
#etc_kwargs = {'dataset_name': 'prueba',
etc_kwargs = {'dataset_name': 'r8-all-terms-clean',
              'initial_step': 1,
              'step_size': 33}
preprocess_kwargs = {'min_word_length': 2,
                     'max_number_words': 1000}
cpi_model_params = dict()
cpi_model_params['alpha'] = 1.0
cpi_model_params['binarize'] = 0.0
cpi_model_params['fit_prior'] = False
cpi_kwargs = {'train_dataset_percentage': 0.75,
              'test_dataset_percentage': 0.25,
              'doc_rep': 'term_frec',
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
etc = EarlyTextClassifier(etc_kwargs, preprocess_kwargs, cpi_kwargs, context_kwargs, dmc_kwargs)
etc.print_params_information()
Xtrain, ytrain, Xtest, ytest = etc.preprocess_dataset()
etc.fit(Xtrain, ytrain)
cpi_percentages, cpi_predictions, dmc_prediction, prediction_time = etc.predict(Xtest, ytest)

performance_kwargs = {'c_tp': 1.0,
                      'c_fn': 1.0,
                      'c_fp': 1.0,
                      'penalization_type': 'Losada-Crestani',
                      'time_threshold': 95}

penalization_type = 'Losada-Crestani'
time_threshold = 95
costs = {'c_tp': 1.0,
         'c_fn': 1.0,
         'c_fp': 1.0}

etc.score(ytest, cpi_predictions, cpi_percentages, prediction_time, penalization_type, time_threshold, costs)
etc.save_model()
