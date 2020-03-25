import matplotlib.pyplot as plt
import numpy as np
from Vowel import Vowel
import global_constants


class Dataset:
    def __init__(self, raw_data, mode=0, samples_to_take_for_each_group=None):  # modes: 0-steady state, 1-20% duration, 2-50%duration, 3-80% duration
        self.vowels = dict(zip(global_constants.vowel_types, [Vowel(vowel) for vowel in global_constants.vowel_types]))  # dictionary with one Vowel class instance for each vowel type

        mode = -1 if len(raw_data[0]) == 4 else mode
        start = 3 + 3 * mode + int(mode != 0) if mode >= 0 else 1

        if samples_to_take_for_each_group is None:
            for sample in reversed(raw_data):
                vowel_type = sample[0][3:5]
                self.vowels[vowel_type].add_point(sample[start:start + 3])
                raw_data.remove(sample)
        else:
            samples_taken = dict(zip(global_constants.vowel_types, [dict(zip(samples_to_take_for_each_group.keys(), np.zeros(len(samples_to_take_for_each_group.keys())))) for i in range(len(global_constants.vowel_types))]))
            for sample in reversed(raw_data):
                speaker_type = sample[0][0]  # m=man, w=woman, b=boy, g=girl
                vowel_type = sample[0][3:5]
                if speaker_type in samples_to_take_for_each_group.keys() and samples_taken[vowel_type][speaker_type] < samples_to_take_for_each_group[speaker_type]:
                    samples_taken[vowel_type][speaker_type] += 1
                    self.vowels[vowel_type].add_point(sample[start:start + 3])
                    raw_data.remove(sample)

    def get_vowel_dict(self):
        return self.vowels

    def classify_point(self, point):
        max_probability = 0
        vowel_type = ''
        for vowel in self.vowels.values():
            probability = vowel.calc_probability(point)
            if probability > max_probability:
                max_probability = probability
                vowel_type = vowel.get_type()
        return vowel_type

    def classify_test_set(self, test_set):
        test_set_vowel_dict = test_set.get_vowel_dict()
        confusion_matrix = dict(zip(test_set_vowel_dict.keys(), [dict(zip(self.vowels.keys(), np.zeros(len(self.vowels.keys()), dtype=int))) for i in range(len(test_set_vowel_dict.keys()))]))
        for vowel_type, vowel in test_set_vowel_dict.items():
            samples = vowel.get_samples()
            for sample in samples:
                confusion_matrix[vowel_type][self.classify_point(sample)] += 1
        return confusion_matrix

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for vowel in self.vowels.values():
            vowel.plot(ax)
        plt.show()
