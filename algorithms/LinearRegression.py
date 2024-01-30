from algorithms.BaseModel import BaseModel
import sklearn.linear_model as lm
import pickle
from utils.save_model import save_model


class LinearRegression(BaseModel):
    def __init__(self):
        self.number_of_drums = None
        self.window = None
        self.model = None

    def create(self, window, number_of_drums):
        """
        Creates a linear regression model.

        :param window: the size of the window of the previous barslices
        :param number_of_drums: the number of drums
        :return: a linear regression model
        """
        self.window = window
        self.number_of_drums = number_of_drums
        self.model = lm.LinearRegression()
        return self.model

    def train(self, X_train, y_train, epochs, model_name='linear_model.sav'):
        """
        Trains a linear regression model.

        :param model: the model to train
        :param X_train: the training data
        :param y_train: the training labels
        :param epochs: the number of epochs to train for
        :param model_path: the path to save the model to
        :return: the trained model
        """
        # TODO convert data to the right format, with windows, instead of the entire sequence
        print(X_train.shape)
        print(y_train.shape)

        self.model.fit(X_train, y_train)

        # Save the model
        save_model(self.model, model_name)

        return self.model

    def evaluate(self, X_test, y_test):
        """
        Evaluates a linear regression model.

        :param model: the model to evaluate
        :param X_test: the test data
        :param y_test: the test labels
        :return: the evaluation results
        """
        return self.model.score(X_test, y_test)
