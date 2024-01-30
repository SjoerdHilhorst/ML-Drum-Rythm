import pickle
import os
from settings import settings


def save_model(model, save_path):
    path = os.path.join(settings['models_dir'], save_path)

    pickle.dump(model, open(path, 'wb'))
    print(f"Model saved to: {path}")
