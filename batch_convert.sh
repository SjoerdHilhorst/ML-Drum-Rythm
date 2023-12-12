#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 input_folder output_folder miditocsv_script"
    exit 1
fi

input_folder="$1"
output_folder="$2"
miditocsv_script="$3"

# Create the output folder if it doesn't exist
mkdir -p "$output_folder"

# Iterate over all MIDI files in the input folder recursively
find "$input_folder" -type f -name "*.midi" -o -name "*.mid" -o -name "*.MID" | while read -r midi_file; do
    # Extract the filename without extension
    file_name=$(basename -- "$midi_file")
    file_name_no_ext="${file_name%.*}"

    # Generate the output CSV file path
    csv_output_file="$output_folder/$file_name_no_ext.csv"

    # Run the miditocsv script
    "./$miditocsv_script" "$midi_file" "$csv_output_file"

    echo "Converted $midi_file to $csv_output_file"
done
