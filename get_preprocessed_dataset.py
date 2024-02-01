import numpy as np


def prepare_data_for_forecasting(data, n_steps, split_value=0.7):
    # Split the data into train and test sets
    train_size = int(len(data) * split_value)
    train_data = data[:train_size]
    test_data = data[train_size:]

    # Prepare train data
    X_train, y_train = [], []
    for i in range(len(train_data) - n_steps):
        X_train.append(train_data[i:(i + n_steps)])
        y_train.append(train_data[i + n_steps])
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Prepare test data
    X_test, y_test = [], []
    for i in range(len(test_data) - n_steps):
        X_test.append(test_data[i:(i + n_steps)])
        y_test.append(test_data[i + n_steps])
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    return X_train, X_test, y_train, y_test


def flatten_timeseries(X):
    return X.reshape((X.shape[0], X.shape[1] * X.shape[2]))


def get_dataset(data_path, timestep_window):
    time_series_data = np.load(data_path).transpose()

    X_train, X_test, y_train, y_test = prepare_data_for_forecasting(time_series_data, timestep_window, split_value=0.7)

    X_train = flatten_timeseries(X_train)
    X_test = flatten_timeseries(X_test)
    return X_train, X_test, y_train, y_test
