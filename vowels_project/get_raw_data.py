import sound_processing
import numpy as np


def from_dat_file(path, duration_mode=0):
    # Modes: 0-steady state, 1-20% duration, 2-50%duration, 3-80% duration
    # Determine what columns to retrieve frequency data from based on mode choice:
    start_col = 3 * (duration_mode + 1) + (duration_mode > 0)
    retrieve = lambda line: [line[0], np.array(line[start_col:start_col + 3])]
    parse = lambda elements: [int(e) if e.isnumeric() else e for e in elements]
    with open(path) as f:
        return [retrieve(parse(line.strip().split())) for line in f.readlines()[1:]]


def normalize(raw_data):
    max_freqs = [freqs.max() for freqs in np.array([sample[1] for sample in raw_data]).transpose()]
    return [[sample[0], sample[1] / max_freqs] for sample in raw_data]
