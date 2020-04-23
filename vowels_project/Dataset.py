import numpy as np
import global_constants


class Dataset:
    def __init__(self, raw_data, samples_to_take_for_each_group=None):
        self.data = {vowel_type: [] for vowel_type in global_constants.vowel_types}  # dictionary with a list of samples for each vowel type
        if samples_to_take_for_each_group is None:  # if the amount samples to take is unspecified, take all samples from the raw data
            for sample in reversed(raw_data):
                vowel_type = sample[0][3:5]
                self.data[vowel_type].append(sample[1])
                raw_data.remove(sample)
        else:  # if the amount of samples to take is specified, take the specified amount
            samples_taken = dict(zip(global_constants.vowel_types, [
                dict(zip(samples_to_take_for_each_group.keys(), np.zeros(len(samples_to_take_for_each_group.keys()))))
                for i in range(len(global_constants.vowel_types))]))
            for sample in reversed(raw_data):
                speaker_type = sample[0][0]  # m=man, w=woman, b=boy, g=girl
                vowel_type = sample[0][3:5]
                if speaker_type in samples_to_take_for_each_group.keys() and samples_taken[vowel_type][speaker_type] < \
                        samples_to_take_for_each_group[speaker_type]:
                    samples_taken[vowel_type][speaker_type] += 1
                    self.data[vowel_type].append(sample[1])
                    raw_data.remove(sample)
        self.data = {vowel_type: np.array(self.data[vowel_type]) for vowel_type in global_constants.vowel_types}  # convert elements in dictionary to numpy arrays
