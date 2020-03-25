import matplotlib.pyplot as plt
import numpy as np
from Vowel import Vowel
import global_constants


class Dataset:
    def __init__(self, raw_data, mode=0, samples_to_take_for_each_group=None):  # modes: 0-steady state, 1-20% duration, 2-50%duration, 3-80% duration
        if samples_to_take_for_each_group is None:
            samples_to_take_for_each_group = {'m': 50, 'w': 50, 'b': 29, 'g': 21}
        self.vowels = dict(zip(global_constants.vowel_types, [Vowel(vowel) for vowel in global_constants.vowel_types]))  # dictionary with one Vowel class instance for each vowel type
        samples_taken = dict(zip(global_constants.vowel_types, [dict(zip(samples_to_take_for_each_group.keys(), np.zeros(len(samples_to_take_for_each_group.keys())))) for i in range(len(global_constants.vowel_types))]))
        for sample in raw_data:
            speaker_type = sample[0][0]  # m=man, w=woman, b=boy, g=girl
            vowel_type = sample[0][3:5]
            if speaker_type in samples_to_take_for_each_group.keys() and samples_taken[vowel_type][speaker_type] < samples_to_take_for_each_group[speaker_type]:
                samples_taken[vowel_type][speaker_type] += 1
                start = 3 + 3 * mode + int(mode != 0)
                self.vowels[vowel_type].add_point(sample[start:start + 3])

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for vowel in self.vowels.values():
            vowel.plot(ax)
        plt.show()
