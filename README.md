# Early Text Classification

This repository contains the implementation of the Early Text Classification framework.

The problem of classification is a widely studied one in supervised learning.
Nonetheless, there are scenarios that received little attention despite its applicability.
One of such scenarios is _early text classification_, where one needs to know the category of a document as soon as possible.
The importance of this variant of the classification problem is evident in tasks like sexual predator detection, where one wants to identify an offender as early as possible.
This framework highlights the two main pieces involved in this problem: _classification with partial information_ and _deciding the
moment of classification_.

Based on the paper:
> Loyola J.M., Errecalde M.L., Escalante H.J., Montes y Gomez M. (2018) Learning When to Classify for Early Text Classification. In: De Giusti A. (eds) Computer Science – CACIC 2017. CACIC 2017. Communications in Computer and Information Science, vol 790. Springer, Cham [[Springer Link]](https://doi.org/10.1007/978-3-319-75214-3_3) [[SEDICI Link]](http://sedici.unlp.edu.ar/handle/10915/63498)

## How to use
The script `test_dataset.py` shows how to use this framework. You need to specify the following parameters:
* etc_kwargs:
    * dataset_name: name of the dataset to use. We expect the dataset to already be splitted in training and test set and to be located inside the folder `dataset`. There should be two files named `{dataset_name}_train.txt` and `{dataset_name}_test.txt`. Each file must have the following structure for each document `i`: `{label_i}[TAB]{document_i}`. Both corpus must end with an empty line.
    * initial_step: initial percentage of the document to read.
    * step_size: percentage of the document to read in each step.
* preprocess_kwargs:
    * min_word_length: number of letters the terms must have to be consider.
    * max_number_words: maximum number of words to consider.
* cpi_kwargs:
    * train_dataset_percentage: cpi train dataset split.
    * test_dataset_percentage: cpi test dataset split.
    * doc_rep: document representation to use. For now the only representation available is `term_frec`.
    * model_type: scikit-learn model to use.
    * cpi_model_params: parameters of the model.
* context_kwargs:
    * number_most_common: number of most common terms of each category to use.
* dmc_kwargs:
    * train_dataset_percentage: dmc train dataset split.
    * test_dataset_percentage: dmc test dataset split.
    * model_type: scikit-learn model to use.
    * dmc_model_params: parameters of the model.

## Dependencies
This code was developed and tested on Python 3.6 and depends on:
* scipy == 1.0.0
* numpy == 1.14.0
* scikit_learn == 0.19.1
 