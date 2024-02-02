import numpy as np


def prepare_data(data, window_size):
    """
    Prepare the data for forecasting by creating a windowed dataset.
    @return: X, y with shapes (n_bar_slices - window_size, window_size, n_drums), (n_bar_slices - window_size, n_drums)
    """
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size)])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)


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


def get_dataset_with_multiple_songs(data_path, window_size, split_value=0.7):
    time_series_data = np.load(data_path).transpose()

    # Get the dimensions of the data (number of bar slices, number of instruments, number of songs)
    num_bar_slices, num_instruments, num_songs = time_series_data.shape

    data_split = int(num_songs * split_value)
    X_train = np.zeros((data_split, (num_bar_slices - window_size), window_size, num_instruments))
    X_test = np.zeros((num_songs - data_split, (num_bar_slices - window_size), window_size, num_instruments))
    y_train = np.zeros((data_split, (num_bar_slices - window_size), num_instruments))
    y_test = np.zeros((num_songs - data_split, (num_bar_slices - window_size), num_instruments))

    # Loop over training data
    for song in range(data_split):
        X_train_song, y_train_song = prepare_data(time_series_data[:, :, song], window_size)

        X_train[song, :, :, :] = X_train_song
        y_train[song, :, :] = y_train_song

    # Loop over testing data
    for song in range(num_songs - data_split):
        X_test_song, y_test_song = prepare_data(time_series_data[:, :, data_split + song], window_size)

        X_test[song, :, :, :] = X_test_song
        y_test[song, :, :] = y_test_song

    X_train = X_train.reshape((X_train.shape[0] * X_train.shape[1], X_train.shape[2] * X_train.shape[3]), order='C')
    X_test = X_test.reshape((X_test.shape[0] * X_test.shape[1], X_test.shape[2] * X_test.shape[3]), order='C')
    y_train = y_train.reshape((y_train.shape[0] * y_train.shape[1], y_train.shape[2]), order='C')
    y_test = y_test.reshape((y_test.shape[0] * y_test.shape[1], y_test.shape[2]), order='C')

    return X_train, X_test, y_train, y_test
