import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    filename = "../img/groove_rock_4drums/linear_regression/trained_model_accuracies/accuracy_probability.txt"
    train_acc, test_acc = [], []

    with open(filename) as f:
        lines = f.readlines()
        train_acc = [float(line.split(" & ")[0]) for line in lines]
        test_acc = [float(line.split(" & ")[1]) for line in lines]

    print(train_acc)
    print(test_acc)

    print("Train accuracy: ", np.mean(train_acc), " +- ", np.std(train_acc))
    print("Test accuracy: ", np.mean(test_acc), " +- ", np.std(test_acc))

