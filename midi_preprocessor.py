import csv
import argparse

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
        drums.add(entry['event_data'][1])

    return drums

def get_ticks_per_bar(midi_data, bar_slices):
    # resolution in midi is done in ticks per quarter note
    # For 4/4 there are 4 beats per bar
    # and each beat is a quarter note (denominator)
    # so the ticks_per_bar are 4 * ticks_per_quarter_note
    # We can now calculate the ticks in a bar slice:
    # ticks_per_bar_slice = ticks_per_par / bar_slices

    # TODO: currently to stupid to handle different time signatures other than 4/4
    # so needs to be implemented, but probably 4/4 is fine for this project

    ticks_per_quarter_note = int(midi_data[0]['event_data'][2])

    time_signature_numerator = int(midi_data[6]['event_data'][0])
    time_signature_denominator = int(midi_data[6]['event_data'][1])

    return ticks_per_quarter_note * time_signature_numerator

def process_midi_data(midi_data, bar_slices=16):
    """
    Process the MIDI data and generate vectors for each bar slice.
    """
    vectors = []

    ticks_per_bar = get_ticks_per_bar(midi_data, bar_slices)
    drums = get_unique_drums(midi_data)

    bar_slice_idx = 0
    bar_slice_data = []
    for entry in midi_data:
        if entry['event_type'] != 'Note_on_c':
            continue
        
        # TODO: set the drum info for a bar slice
        pass

        vectors.append(bar_slice_data)

    return vectors

def main():
    parser = argparse.ArgumentParser(description="Process MIDI information from a CSV file.")
    parser.add_argument("input_file", help="Path to the input CSV file containing MIDI information.")
    parser.add_argument("--bar_slices", type=int, default=16, help="Number of slices in one bar. Default is 16.")
    args = parser.parse_args()

    midi_data = parse_csv(args.input_file)
    vectors = process_midi_data(midi_data, bar_slices=args.bar_slices)

    # print("Output Vectors:")
    # for vector in vectors:
    #     print(vector)

if __name__ == "__main__":
    main()
