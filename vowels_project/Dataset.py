import matplotlib.pyplot as plt
import numpy as np
from Vowel import Vowel
import global_constants


class Dataset:
    def __init__(self, raw_data, mode=0, samples_to_take_for_each_group=None, n_components=1, covariance_type='full'):  # modes: 0-steady state, 1-20% duration, 2-50%duration, 3-80% duration
        self.vowels = dict(zip(global_constants.vowel_types, [Vowel(vowel, n_components, covariance_type=covariance_type) for vowel in global_constants.vowel_types]))  # dictionary with one Vowel class instance for each vowel type

        mode = -1 if len(raw_data[0]) == 4 else mode  # if raw_data is from .wav analysis there is only one set of points to choose
        start_col = 3 + 3 * mode + int(mode != 0) if mode >= 0 else 1  # determine what columns to retrieve frequency data from based on mode choice

        if samples_to_take_for_each_group is None:  # if the amount samples to take is unspecified, take all samples from the raw data
            for sample in reversed(raw_data):
                vowel_type = sample[0][3:5]
                self.vowels[vowel_type].add_point(sample[start_col:start_col + 3])
                raw_data.remove(sample)
        else:  # if the amount of samples to take is specified, take the specified amount
            samples_taken = dict(zip(global_constants.vowel_types, [dict(zip(samples_to_take_for_each_group.keys(), np.zeros(len(samples_to_take_for_each_group.keys())))) for i in range(len(global_constants.vowel_types))]))
            for sample in reversed(raw_data):
                speaker_type = sample[0][0]  # m=man, w=woman, b=boy, g=girl
                vowel_type = sample[0][3:5]
                if speaker_type in samples_to_take_for_each_group.keys() and samples_taken[vowel_type][speaker_type] < samples_to_take_for_each_group[speaker_type]:
                    samples_taken[vowel_type][speaker_type] += 1
                    self.vowels[vowel_type].add_point(sample[start_col:start_col + 3])
                    raw_data.remove(sample)

    def get_vowel_dict(self):
        return self.vowels

    def train(self):  # based on the samples in the dataset, fit a gaussian probability distribution
        for vowel in self.vowels.values():
            vowel.samples = np.array(vowel.samples)
            vowel.fit_gmm()

    def classify_point(self, point):  # for every vowel in the dataset, check the probability that the point is from the vowels distribution. The result is the vowel with the highest probability
        max_score = -1000
        vowel_type = ''
        for vowel in self.vowels.values():
            score = vowel.score(point)
            if score > max_score:
                max_score = score
                vowel_type = vowel.get_type()
        return vowel_type

    def classify_test_set(self, test_set):  # classify all points in the test set using the classify_point method
        test_set_vowel_dict = test_set.get_vowel_dict()
        confusion_matrix = np.zeros((len(global_constants.vowel_types), len(global_constants.vowel_types)), dtype=int)
        vowel_type_to_index = dict(zip(global_constants.vowel_types, np.arange(len(global_constants.vowel_types))))
        total, correct = 0, 0
        for vowel_type, vowel in test_set_vowel_dict.items():  # for all vowel types in the test set
            samples = vowel.get_samples()  # get the samples
            for sample in samples:  # for all samples
                classify_result = self.classify_point(sample)  # classify the sample
                confusion_matrix[vowel_type_to_index[classify_result]][vowel_type_to_index[vowel_type]] += 1  # increment the appropriate  cell in the confusion matrix
                total += 1
                correct += 1 if classify_result == vowel_type else 0
        return 1 - correct / total, confusion_matrix

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for vowel in self.vowels.values():
            vowel.plot(ax)
        plt.show()
