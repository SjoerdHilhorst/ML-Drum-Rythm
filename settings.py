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
        # 37: "Side Stick",
        38: "Acoustic Snare",
        # 40: "Electric Snare",
        42: "Closed Hi-Hat",
        # 44: "Pedal Hi-Hat",
        # 45: "Low Tom",
        # 46: "Open Hi-Hat",
        # 48: "Hi-Mid Tom",
        49: "Crash Cymbal 1",
        # 50: "High Tom",
        # 51: "Ride Cymbal 1",
        # 52: "Chinese Cymbal",
        # 53: "Ride Bell",
        # 55: "Splash Cymbal",
        # 57: "Crash Cymbal 2",
        # 58: "Vibraslap",
        # 59: "Ride Cymbal 2",
        # 56: "Cowbell",
    },

    # how many bar slices to generate in the final beat
    "slices_to_generate": 64,

    # path to the folder where the models are saved
    "models_dir": "models",

    # settings for the decision-making algorithms
    "scaling_factor": 4.0,
    "threshold": 0.05,
}