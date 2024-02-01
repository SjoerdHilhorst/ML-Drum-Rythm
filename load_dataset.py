import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os
from midiutil import MIDIFile

def write_midi_from_string(midi_content, output_file):
    midi_file = MIDIFile()
    midi_file.open_string(midi_content)

    with open(output_file, 'wb') as file:
        midi_file.writeFile(file)


def filter_examples(example):
    return tf.math.logical_and(
        tf.equal(example['style']['primary'], 16),
        tf.equal(example['time_signature'], 1)
    )
    
if __name__ == "__main__":
    tf.compat.v1.enable_eager_execution()
    directory_path = os.path.dirname("./rock-44-time/")
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    dataset, info = tfds.load('groove/4bar-midionly', split='train', with_info=True)
    filtered_dataset = dataset.filter(filter_examples)
    
    i = 0
    for example in filtered_dataset:
        example_midi_tensor = example['midi']
        midi_content_bytes = example_midi_tensor.numpy()
        output_file =   './rock-44-time/example-%d.mid' % i

        with open(output_file, 'wb') as file:
            file.write(midi_content_bytes)

        i+=1

    print("%d examples written" % i)
