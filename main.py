from midi_preprocessor import preprocess
from model import main as train_model
from beat_generator import main as generate_beats


if __name__ == "__main__":
    # Obtain dataset
    preprocess("drum_patterns_csv/2ROCK.csv")

    # Train model
    train_model()

    # Generate beats
    generate_beats()


