class ContextInformation:
    training_data_information = None

    def __init__(self, context_kwargs):
        print("Creando clase ContextInformation con los siguientes parámetros:")
        print(context_kwargs)

    def get_training_information(self, Xtrain=None, ytrain=None):
        print("Obtaining information from the preprocessed training data")
        if (Xtrain is None) and (ytrain is None):
            self.training_data_information = None
        return self.training_data_information

    def generate_dmc_dataset(self, cpi_Xtest, cpi_ytest, cpi_prediction, training_data_information, dmc_kwargs):
        print("Generating DecisionClassifier dataset")
        return None, None

    def test(self):
        print("Nothing")