from midi_preprocessor import preprocess
from model import train_model
from beat_generator import generate_beats
from decision_making import *


if __name__ == "__main__":
    model_path = 'linear_model.sav'
    data_path = 'data/groove_rock.npy'
    initial_drumbeat = 4000

    # Obtain dataset
    # preprocess("drum_patterns_csv/2ROCK.csv")

    # Train model
    # train_model(model_path, data_path=data_path)

    # Get a simple sequence to start synthesis
    data = np.load(data_path).transpose()
    initial_slices = data[:settings["window"], :, initial_drumbeat].flatten()

    # Generate beats
    for decision_making_algorithm in [threshold_signal, probability_signal, combined_decision_algorithm]:
        generate_beats(model_path,
                       decision_algorithm=decision_making_algorithm,
                       save_path="img/linear_regression/initial_drumbeat" + str(initial_drumbeat) + "/",
                       initial_slices=initial_slices)
