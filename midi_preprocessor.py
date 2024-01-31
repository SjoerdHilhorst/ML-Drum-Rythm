import csv
import argparse
from pprint import pprint
from settings import settings
import numpy as np

def parse_csv(file_path):
    """
    Parse the CSV file containing MIDI information.
    """
    midi_data = []
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            track = int(row[0])
            tick = int(row[1])
            event_type = row[2].strip()
            event_data = row[3:]

            midi_data.append({'track': track, 'tick': tick, 'event_type': event_type, 'event_data': event_data})

    return midi_data

def get_unique_drums(midi_data):
    """
    Obtain unique drums from midi_data
    A drum corresponds with a note in the Note_on_c event
    """
    drums = set()
    for entry in midi_data:
        if entry['event_type'] != 'Note_on_c':
            continue
        drums.add(int(entry['event_data'][1]))
    return drums

def get_ticks_per_bar(midi_data, bar_slices):
    # resolution in midi is done in ticks per quarter note
    # For 4/4 there are 4 beats per bar
    # and each beat is a quarter note (denominator)
    # so the ticks_per_bar are 4 * ticks_per_quarter_note
    # We can now calculate the ticks in a bar slice:
    # ticks_per_bar_slice = ticks_per_par / bar_slices
    # because for this project we only use 4/4 we can simply do
    # ticks_per_quarter_note * 4

    ticks_per_quarter_note = int(midi_data[0]['event_data'][2])

    return ticks_per_quarter_note * 4


def get_total_bar_slices(midi_data, ticks_per_bar, bar_slices):
    ticks_per_quarter_note = int(midi_data[0]['event_data'][2])

    return bar_slices * midi_data[-2]['tick'] // ticks_per_bar

def process_midi_data(midi_data, bar_slices=16, use_velocity=False):
    """
    Process the MIDI data and generate vectors for each bar slice.
    """
    vectors = []

    ticks_per_bar = get_ticks_per_bar(midi_data, bar_slices)
    total_bar_slices = get_total_bar_slices(midi_data, ticks_per_bar, bar_slices)

    drums = list(settings["midi_notes"].keys())

    bar_slice_data = {drum: [0] * total_bar_slices for drum in drums}

    for entry in midi_data:
        if entry['event_type'] != 'Note_on_c':
            continue
        
        tick = entry['tick']
        _, note, velocity = [int(value) for value in entry['event_data']]

        if velocity == 0:
            continue
        
        # only process notes that are defined in midi_notes_to_sounds
        if note not in bar_slice_data:
            continue
        
        bar_slice_idx = tick // (ticks_per_bar // bar_slices)

        # Update the velocity information in bar_slice_data
        bar_slice_data[note][bar_slice_idx] = velocity if use_velocity else 1

    # Convert bar_slice_data to a list of vectors
    for drum, velocities in bar_slice_data.items():
        vectors.append(velocities)

    return vectors

def main():
    parser = argparse.ArgumentParser(description="Process MIDI information from a CSV file.")
    parser.add_argument("input_file", help="Path to the input CSV file containing MIDI information.")
    parser.add_argument("--bar_slices", type=int, default=settings["bar_slices"], help="Number of slices in one bar. Default is 16.")
    parser.add_argument("--use_velocity", action="store_true", default=settings["use_velocity"], help="Use velocity to encode drum in vector. Default is false (use 1 or 0)")
    args = parser.parse_args()

    midi_data = parse_csv(args.input_file)
    vectors = process_midi_data(midi_data, bar_slices=args.bar_slices, use_velocity=args.use_velocity)

    np.save("rockdata.npy", vectors)
    print("Saved drum data to data.npy!")


def preprocess(input_file, bar_slices=16, use_velocity=False):
    """
    Preprocess the MIDI data and generate vectors for each bar slice.
    Alternative to main() for use in other files.
    """
    midi_data = parse_csv(input_file)
    vectors = process_midi_data(midi_data, bar_slices=bar_slices, use_velocity=use_velocity)

    np.save("data.npy", vectors)
    print("Saved drum data to data.npy!")


if __name__ == "__main__":
    main()
