import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import argparse
from settings import settings
import keras

def create_mlp_model(timestep_window, number_of_drums):
    model = Sequential()

    model.add(Dense(64, activation='relu', input_dim=(timestep_window * number_of_drums)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(number_of_drums, activation='softmax'))

    return model

def train_model(model, X_train, y_train, epochs, save_path):
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'], )
    model.fit(X_train, y_train, epochs=epochs)

    model.save(save_path)
    print(f"Model saved to: {save_path}")

def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test)

    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")

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


def main():
    parser = argparse.ArgumentParser(description="Train and save/load a TensorFlow model.")
    parser.add_argument('--model_path', default='my_model.keras', help="Path to save/load the model")
    parser.add_argument('--window', default=settings["window"], help="the size of the window of the previous barslices")
    parser.add_argument('--data_path', default='data.npy', help="Path to the dataset")
    parser.add_argument('--epochs', default=settings["epochs"], help="number of training epochs")
    
    args = parser.parse_args()

    # Obtain dataset
    X_train, X_test, y_train, y_test = get_dataset(args.data_path, args.window)

    # Training model
    print("Training model...")
    number_of_drums = len(settings["midi_notes"])
    model = create_mlp_model(args.window, number_of_drums)
    train_model(model, X_train, y_train, int(args.epochs), args.model_path)

    # Evaluating model
    print("Evaluating model...")
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()