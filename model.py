import argparse
import os

from algorithms.LinearRegression import LinearRegression
from settings import settings
from get_preprocessed_dataset import get_dataset_with_multiple_songs
from decision_making import *


def train_model(model_name="my_model.keras",
                window=settings["window"],
                data_path="data.npy",
                save_path="img/unspecified/",
                epochs=settings["epochs"],
                decision_algorithm=None,
                filename="accuracies.txt"):
    # Obtain dataset
    X_train, X_test, y_train, y_test = get_dataset_with_multiple_songs(data_path, window)

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
    if decision_algorithm is None:
        print(algorithm.evaluate(X_test, y_test))
    else:
        train_accuracy = algorithm.evaluate_with_decision_algorithm(X_train, y_train, decision_algorithm)
        test_accuracy = algorithm.evaluate_with_decision_algorithm(X_test, y_test, decision_algorithm)

        # Save accuracies to file
        if decision_algorithm == threshold_signal:
            if not os.path.exists(save_path + filename):
                # Create new file and write
                with open(save_path + filename, "w") as file:
                    file.write("Threshold & Train accuracy & Test accuracy\n")
                    file.write("{:.1f}".format(settings["threshold"]) + " & " + "{:.3f}".format(train_accuracy) + " & " + "{:.3f}".format(test_accuracy) + "\n")
            else:
                # Append to file
                with open(save_path + filename, "a") as file:
                    file.write("{:.1f}".format(settings["threshold"]) + " & " + "{:.3f}".format(train_accuracy) + " & " + "{:.3f}".format(test_accuracy) + "\n")

        if decision_algorithm == combined_decision_algorithm:
            if not os.path.exists(save_path + filename):
                # Create new file and write
                with open(save_path + filename, "w") as file:
                    file.write("Scaling factor & Train accuracy & Test accuracy\n")
                    file.write("{:.1f}".format(settings["scaling_factor"]) + " & " + "{:.3f}".format(train_accuracy) + " & " + "{:.3f}".format(test_accuracy) + "\n")
            else:
                # Append to file
                with open(save_path + filename, "a") as file:
                    file.write("{:.1f}".format(settings["scaling_factor"]) + " & " + "{:.3f}".format(train_accuracy) + " & " + "{:.3f}".format(test_accuracy) + "\n")

        if decision_algorithm == probability_signal:
            if not os.path.exists(save_path + filename):
                # Create new file and write
                with open(save_path + filename, "w") as file:
                    file.write("{:.3f}".format(train_accuracy) + " & " + "{:.3f}".format(test_accuracy) + "\n")
            else:
                # Append to file
                with open(save_path + filename, "a") as file:
                    file.write("{:.3f}".format(train_accuracy) + " & " + "{:.3f}".format(test_accuracy) + "\n")


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