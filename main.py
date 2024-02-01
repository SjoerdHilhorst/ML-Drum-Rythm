from midi_preprocessor import preprocess
from model import train_model
from beat_generator import generate_beats
from decision_making import *


if __name__ == "__main__":
    model_path = 'linear_model.sav'
    data_path = 'rockdata.npy'
    decision_making_algorithm = threshold_signal

    # Obtain dataset
    # preprocess("drum_patterns_csv/2ROCK.csv")

    # Train model
    # train_model(model_path, data_path=data_path)

    # Generate beats
    generate_beats(model_path,
                   decision_algorithm=decision_making_algorithm)
