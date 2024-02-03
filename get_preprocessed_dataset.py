import numpy as np
from beat_generator import plot as plot_beat

def prepare_data_for_forecasting(data, n_steps, split_value=0.7):
    # data shape: (number_of_timesteps, number_of_drums, number_of_examples)
    number_of_examples = data.shape[2]
    train_size = int(number_of_examples * split_value)
    
    # Initialize lists for the training and testing sets
    X_train, y_train = [], []
    X_test, y_test = [], []

    # Function to create datasets
    def create_dataset(data, start, end):
        X, y = [], []
        for i in range(start, end):
            for j in range(data.shape[0] - n_steps):
                X.append(data[j:(j + n_steps), :, i])
                y.append(data[j + n_steps, :, i])
        return np.array(X), np.array(y)

    # Prepare training and testing datasets
    X_train, y_train = create_dataset(data, 0, train_size)
    X_test, y_test = create_dataset(data, train_size, number_of_examples)
    
    return X_train, X_test, y_train, y_test


def flatten_timeseries(X):
    return X.reshape((X.shape[0], X.shape[1] * X.shape[2]))


def get_dataset(data_path, timestep_window):
    time_series_data = np.load(data_path).transpose()

    X_train, X_test, y_train, y_test = prepare_data_for_forecasting(time_series_data, timestep_window, split_value=0.7)

    X_train = flatten_timeseries(X_train)
    X_test = flatten_timeseries(X_test)
    return X_train, X_test, y_train, y_test
