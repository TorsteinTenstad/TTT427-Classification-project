import sound_processing
import numpy as np
# ====================================================================
#       File format
# ====================================================================
# Filenames:
# character 1:     m=man, w=woman, b=boy, g=girl
# characters 2-3:  talker number
# characters 4-5:  vowel (ae="had", ah="hod", aw="hawed", eh="head", er="heard",
#                        ei="haid", ih="hid", iy="heed", oa=/o/ as in "boat",
#                        oo="hood", uh="hud", uw="who'd")
#
# col1:  filename, #col2:  duration in msec, #col3:  f0 at "steady state", #col4:  F1 at "steady state", #col5:  F2
# at "steady state" col6:  F3 at "steady state", #col7:  F4 at "steady state", #col8:  F1 at 20% of vowel duration,
# #col9:  F2 at 20% of vowel duration col10: F3 at 20% of vowel duration, #col11: F1 at 50% of vowel duration,
# #col12: F2 at 50% of vowel duration, #col13: F3 at 50% of vowel duration col14: F1 at 80% of vowel duration,
# #col15: F2 at 80% of vowel duration, #col16: F3 at 80% of vowel duration
#
# Note: An entry of zero means that the formant was not measurable.


def from_dat_file(path, mode=0):  # modes: 0-steady state, 1-20% duration, 2-50%duration, 3-80% duration
    start_col = 3 * (mode + 1) + (mode > 0)  # determine what columns to retrieve frequency data from based on mode choice
    retrieve = lambda line: [line[0], np.array(line[start_col:start_col + 3])]
    parse = lambda elements: [int(e) if e.isnumeric() else e for e in elements]
    with open(path) as f:
        return [retrieve(parse(line.strip().split())) for line in f.readlines()[1:]]


def from_wav_analysis(folder):
    raw_data = []
    for subfolder in ['men', 'women', 'kids']:
        raw_data = raw_data + sound_processing.get_raw_feature_data(folder + '/' + subfolder)
    return raw_data


def normalize(raw_data):
    max_freqs = [freqs.max() for freqs in np.array([sample[1] for sample in raw_data]).transpose()]
    return [[sample[0], sample[1] / max_freqs] for sample in raw_data]


