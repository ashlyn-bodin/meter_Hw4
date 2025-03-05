from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.metrics import accuracy_score

# Importing the parent: DataPreprocessing class from data_preprocess.py
from src.data_preprocess import DataPreprocessing 


class ModelBuilder(DataPreprocessing):
    def __init__(self, *args, **kwargs):
        super(ModelBuilder, self).__init__(*args, **kwargs)

    def dt(self, X_train, X_test, y_train, y_test):
        #Create DT model
        ANN_classifier = MLPClassifier()

        #Train the model
        ANN_classifier.fit(X_train, y_train)

        #Test the model
        ANN_predicted = ANN_classifier.predict(X_test)

        error = 0
        for i in range(len(y_test)):
            error += np.sum(ANN_predicted != y_test)

        total_accuracy = 1 - error / len(y_test)

        #get performance
        self.accuracy = accuracy_score(y_test, ANN_predicted)

        return ANN_classifier