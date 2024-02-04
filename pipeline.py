from midi_preprocessor import preprocess
from model import train_model
from beat_generator import generate_beats
from decision_making import *


if __name__ == "__main__":
    model_path = 'linear_model.sav'
    initial_drumbeat = 71
    data_path = 'data/groove_rock_4drums.npy'
    save_path = "img/groove_rock_4drums/linear_regression/initial_drumbeat" + str(initial_drumbeat) + "/"

    # Train model
    train_model(model_path, data_path=data_path, save_path="img/groove_rock_4drums/linear_regression/trained_model_accuracies/")

    # Alternative: vary over threshold values
    # for threshold in np.linspace(0.1, 0.9, 9):
    #     settings['threshold'] = threshold
    #     train_model(model_path, data_path=data_path, decision_algorithm=threshold_signal,
    #                 filename="accuracy_threshold.txt",
    #                 save_path="img/groove_rock_4drums/linear_regression/trained_model_accuracies/")

    # Alternative: vary over scaling factor
    # for scaling_factor in np.linspace(1.0, 20.0, 20):
    #     settings['scaling_factor'] = scaling_factor
    #     train_model(model_path, data_path=data_path, decision_algorithm=combined_decision_algorithm,
    #                 filename="accuracy_scaling_factor.txt",
    #                 save_path="img/groove_rock_4drums/linear_regression/trained_model_accuracies/")

    # Alternative: repeat non-deterministic decision-making algorithm
    # for _ in range(100):
    #     train_model(model_path, data_path=data_path, decision_algorithm=probability_signal,
    #                 filename="accuracy_probability.txt",
    #                 save_path="img/groove_rock_4drums/linear_regression/trained_model_accuracies/")

    # Get a simple sequence to start synthesis
    data = np.load(data_path).transpose()
    initial_slices = data[:settings["window"], :, initial_drumbeat].flatten()

    # Generate beats
    # for decision_making_algorithm in [threshold_signal, probability_signal, combined_decision_algorithm]:
    #     generate_beats(model_path,
    #                    decision_algorithm=decision_making_algorithm,
    #                    save_path=save_path,
    #                    initial_slices=initial_slices)

    # # Alternative: vary over threshold values
    for threshold in [0.1, 0.25, 0.5, 0.75, 0.9]:
        settings['threshold'] = threshold
        generate_beats(model_path,
                       decision_algorithm=threshold_signal,
                       save_path=save_path,
                       initial_slices=initial_slices)

    # Alternative: vary over scaling factor
    for scaling_factor in [1.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20.0]:
        settings['scaling_factor'] = scaling_factor
        generate_beats(model_path,
                       decision_algorithm=combined_decision_algorithm,
                       save_path=save_path,
                       initial_slices=initial_slices)
