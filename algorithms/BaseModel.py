import numpy as np


class BaseModel:
    """
    Base class for both models: MLP and linear regression.
    """
    def create(self, window, number_of_drums):
        raise NotImplementedError

    def train(self, X_train, y_train, epochs, model_path):
        raise NotImplementedError

    def evaluate(self, X_test, y_test):
        raise NotImplementedError
