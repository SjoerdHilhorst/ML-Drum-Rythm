# in this file you can determine which drum sounds you want in the dataset
# https://www.zendrum.com/resource-site/drumnotes.htm
settings = {
    # how many previous timesteps are used
    "window": 16,

    # use integer velocity values or boolean
    "use_velocity": False,

    # how many slices are in one bar, default 16
    "bar_slices": 16,

    # how many training epochs
    "epochs": 100,

    # determine which instruments are used
    "midi_notes": {
        36: "Bass Drum 1",
        # 39: "Hand Clap",
        40: "Electric Snare",
        42: "Closed Hi-Hat",
        # 44: "Pedal Hi-Hat",
        # 45: "Low Tom 1",
        # 46: "Hi-Hat Open",
        # 49: "Crash Cymbal 1",
        # 51: "Ride Cymbal 1",
        # 53: "Ride Bell",
        56: "Cowbell",
    },

    # how many bar slices to generate in the final beat
    "slices_to_generate": 64 

}