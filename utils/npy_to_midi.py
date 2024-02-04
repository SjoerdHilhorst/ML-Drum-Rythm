import numpy as np
from mido import MidiFile, MidiTrack, Message


def npy_to_midi(npy_file, midi_file):
    """
    Convert a .npy file to a .midi file.
    """
    # Load the .npy file
    slices = np.load(npy_file)

    # Create a new MIDI file
    midi = MidiFile()
    track = MidiTrack()
    midi.tracks.append(track)

    # Add the slices to the MIDI file
    for slice in slices:
        for i, note in enumerate(slice):
            if note == 1:
                # Add note on event
                track.append(Message('note_on', note=i, velocity=64, time=0))

                # Add note off event
                track.append(Message('note_off', note=i, velocity=64, time=96))

    # Save the MIDI file
    midi.save(midi_file)

    # def array2midi(arr, tempo=500000):
    #     """Converts a numpy array to a MIDI file"""
    #     # Adapted from: https://medium.com/analytics-vidhya/convert-midi-file-to-numpy-array-in-python-7d00531890c
    #     new_arr = np.concatenate([np.array([[0] * 128]), np.array(arr)], axis=0)
    #     changes = new_arr[1:] - new_arr[:-1]
    #     midi_file = mido.MidiFile()  # create a midi file with an empty track
    #     track = mido.MidiTrack()
    #     midi_file.tracks.append(track)
    #     track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))
    #     last_time = 0
    #     for ch in changes:  # add difference in the empty track
    #         if set(ch) == {0}:  # no change
    #             last_time += 1
    #         else:
    #             on_notes = np.where(ch > 0)[0]
    #             on_notes_vol = ch[on_notes]
    #             off_notes = np.where(ch < 0)[0]
    #             first_ = True
    #             for n, v in zip(on_notes, on_notes_vol):
    #                 new_time = last_time if first_ else 0
    #                 track.append(mido.Message('note_on', note=n, velocity=v, time=new_time))
    #                 first_ = False
    #             for n in off_notes:
    #                 new_time = last_time if first_ else 0
    #                 track.append(mido.Message('note_off', note=n, velocity=0, time=new_time))
    #                 first_ = False
    #             last_time = 0
    #     return midi_file


if __name__ == "__main__":
    track = np.load("../img/groove_rock_4drums/linear_regression/initial_drumbeat0/generated_threshold_0.5.npy")
    print(track.shape)

    npy_to_midi(track, "../img/groove_rock_4drums/linear_regression/initial_drumbeat0/generated_threshold_0.5.mid")
