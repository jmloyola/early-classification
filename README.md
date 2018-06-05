# early-classification
**Early Text Classification**

This repository contains the implementation of the Early Text Classification framework described in this [paper](https://doi.org/10.1007/978-3-319-75214-3_3).

The script `test_dataset.py` shows how to use this framework. You need to specify the following parameters:
* etc_kwargs:
    * dataset_name:
    * initial_step:
    * step_size:
* preprocess_kwargs:
    * aa:
* cpi_kwargs:
    * train_dataset_percentage:
    * test_dataset_percentage: 
    * doc_rep:
    * model_type:
    * cpi_model_params:
* context_kwargs:
    * number_most_common:
* dmc_kwargs:
    * train_dataset_percentage:
    * test_dataset_percentage:
    * model_type:
    * dmc_model_params:
* performance_kwargs:
    * aa: