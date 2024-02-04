from midi_preprocessor import preprocess
from model import train_model
from beat_generator import generate_beats


if __name__ == "__main__":
    model_path = 'linear_model.sav'

    # Obtain dataset
    preprocess("rock-44-time-csv")

    # Train model
    train_model(model_path)

    # Generate beats
    generate_beats(model_path)
