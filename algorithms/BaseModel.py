import numpy as np


class BaseModel:
    """
    Base class for both models: MLP and linear regression.
    """
    def create_model(self, window, number_of_drums):
        raise NotImplementedError

    def train_model(self, model, X_train, y_train, epochs, model_path):
        raise NotImplementedError

    def evaluate_model(self, model, X_test, y_test):
        raise NotImplementedError

    def prepare_data_for_forecasting(self, data, n_steps, split_value=0.7):
        raise NotImplementedError

    def flatten_timeseries(self, X):
        return X.reshape((X.shape[0], X.shape[1] * X.shape[2]))

    def get_dataset(self, data_path, timestep_window):
        time_series_data = np.load(data_path).transpose()

        X_train, X_test, y_train, y_test = self.prepare_data_for_forecasting(time_series_data, timestep_window,
                                                                             split_value=0.7)

        X_train = self.flatten_timeseries(X_train)
        X_test = self.flatten_timeseries(X_test)
        return X_train, X_test, y_train, y_test
