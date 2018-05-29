class ContextInformation:
    def __init__(self, context_kwargs):
        print("Creando clase ContextInformation con los siguientes parámetros:")
        print(context_kwargs)

    def get_training_information(self, Xtrain, ytrain):
        print("Obtaining information from the preprocessed training data")
        return None

    def generate_dmc_dataset(self, cpi_Xtest, cpi_ytest, cpi_prediction, training_data_information):
        print("Generating DecisionClassifier dataset")
        return None, None

    def test(self):
        print("Nothing")