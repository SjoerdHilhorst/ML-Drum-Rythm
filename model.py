import argparse
from algorithms.LinearRegression import LinearRegression
from settings import settings
from get_preprocessed_dataset import get_dataset


def train_model(model_name="my_model.keras", window=settings["window"], data_path="data.npy", epochs=settings["epochs"]):
    # Obtain dataset
    X_train, X_test, y_train, y_test = get_dataset(data_path, window)

    # Choosing model
    print("Creating model...")
    algorithm = LinearRegression()

    # Training model
    print("Training model...")
    number_of_drums = len(settings["midi_notes"])
    algorithm.create(window, number_of_drums)
    algorithm.train(X_train, y_train, int(epochs), model_name)

    # Evaluating model
    print("Evaluating model...")
    print(algorithm.evaluate(X_test, y_test))


def main():
    parser = argparse.ArgumentParser(description="Train and save/load a TensorFlow model.")
    parser.add_argument('--model_path', default='my_model.keras', help="Path to save/load the model")
    parser.add_argument('--window', default=settings["window"], help="the size of the window of the previous barslices")
    parser.add_argument('--data_path', default='rockdata.npy', help="Path to the dataset")
    parser.add_argument('--epochs', default=settings["epochs"], help="number of training epochs")
    
    args = parser.parse_args()

    train_model(args.model_path, args.window, args.data_path, args.epochs)


if __name__ == "__main__":
    main()