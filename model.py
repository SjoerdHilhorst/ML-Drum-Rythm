import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def create_mlp_model(n, k):
    model = Sequential()

    model.add(Dense(64, activation='relu', input_dim=(n * k)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(k, activation='softmax'))

    return model

def train_model(model, X_train, y_train, epochs):
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs)

def generate_predictions(model, X_test):
    predictions = model.predict(X_test)
    return predictions

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

def flattten_timeseries(X):
    return X.reshape((X.shape[0], X.shape[1] * X.shape[2]))


if __name__ == "__main__":
    n = 8  # Number of previous timesteps

    # Example data generation (replace this with your time series data)
    time_series_data = np.load("data.npy").transpose()  # Replace with your actual
    print(time_series_data.shape)
    k = time_series_data.shape[1]


    # Create MLP model
    model = create_mlp_model(n, k)
    X_train, X_test, y_train, y_test = prepare_data_for_forecasting(time_series_data, n, split_value=0.7)

    X_train = flattten_timeseries(X_train)
    X_test = flattten_timeseries(X_test)

    # Train the model
    epochs = 30  # Number of training epochs
    train_model(model, X_train, y_train, epochs)

    loss, accuracy = model.evaluate(X_test, y_test)

    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")

    outcome = model.predict(X_test)
    outcome = np.where(outcome > 0.2, 1, 0)

    print(outcome, y_test)