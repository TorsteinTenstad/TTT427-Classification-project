import numpy as np
import matplotlib.pyplot as plt
import global_constants


class Dataset:
    def __init__(self, raw_data, samples_to_take_for_each_group=None):
        # Dictionary with a list of samples for each vowel type:
        self.data = {vowel_type: [] for vowel_type in global_constants.vowel_types}
        # If the amount samples to take is unspecified, take all samples from the raw data:
        if samples_to_take_for_each_group is None:
            for sample in reversed(raw_data):
                vowel_type = sample[0][3:5]
                self.data[vowel_type].append(sample[1])
                raw_data.remove(sample)
        # If the amount of samples to take is specified, take the specified amount:
        else:
            samples_taken = dict(zip(global_constants.vowel_types, [dict(
                zip(samples_to_take_for_each_group.keys(),
                    np.zeros(len(samples_to_take_for_each_group.keys())))) for i in
                range(len(global_constants.vowel_types))]))
            for sample in reversed(raw_data):
                speaker_type = sample[0][0]  # m=man, w=woman, b=boy, g=girl
                vowel_type = sample[0][3:5]
                if speaker_type in samples_to_take_for_each_group.keys() and \
                        samples_taken[vowel_type][speaker_type] < samples_to_take_for_each_group[
                            speaker_type]:
                    samples_taken[vowel_type][speaker_type] += 1
                    self.data[vowel_type].append(sample[1])
                    raw_data.remove(sample)
        self.data = {vowel_type: np.array(self.data[vowel_type]) for vowel_type in
                     global_constants.vowel_types}  # convert elements in dictionary to numpy arrays

    def plot(self):
        fig, axs = plt.subplots(3)
        axis_labels = [['2', '3'], ['1', '3'], ['1', '2']]
        for subplot in range(len(axs)):
            for vowel_type, datapoints in self.data.items():
                datapoints = np.delete(datapoints, subplot, axis=1)
                axs[subplot].scatter(datapoints[:, 0], datapoints[:, 1],
                                     c=global_constants.colors[vowel_type], label=vowel_type, s=10)
                axs[subplot].set(xlabel='Normalized F' + axis_labels[subplot][0],
                                 ylabel='Normalized F' + axis_labels[subplot][1])
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=4)
        fig.set_size_inches(4.77, 8.19)
        fig.tight_layout()
        plt.show()
