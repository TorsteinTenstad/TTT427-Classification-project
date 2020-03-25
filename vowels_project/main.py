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

import numpy as np
import matplotlib.pyplot as plt

vowel_types = ['ae', 'ah', 'aw', 'ae', 'ah', 'aw', 'eh', 'er', 'ei', 'ih', 'iy', 'oa', 'oo', 'uh', 'uw']

colors = {"ae": "black",
          "ah": "dimgrey",
          "aw": "silver",
          "eh": "lightcoral",
          "ei": "firebrick",
          "er": "crimson",
          "ih": "darkorange",
          "iy": "yellow",
          "oa": "lime",
          "oo": "darkgreen",
          "uh": "cornflowerblue",
          "uw": "mediumblue"}


class Dataset:
    def __init__(self, raw_data, take_every_x_sample, mode):  # modes: 0-steady state, 1-20% duration, 2-50%duration, 3-80% duration
        self.vowels = dict(zip(vowel_types, [Vowel(vowel) for vowel in vowel_types]))
        for sample in raw_data:
            if not int(sample[0][1:3]) % take_every_x_sample:
                start = 3 + 3 * mode + int(mode != 0)
                self.vowels[sample[0][3:5]].add_point(sample[start:start + 3])

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for vowel in self.vowels.values():
            vowel.plot(ax)
        plt.show()


class Vowel:
    def __init__(self, vowel_type):
        self.type = vowel_type
        self.points = []
        self.mean = 0
        self.covariance = []

    def calc_mean(self):
        self.mean = np.average(self.points, axis=0)

    def calc_covariance_matrix(self):
        self.covariance = np.cov(np.transpose(self.points))

    def add_point(self, point):
        self.points.append(point)

    def get_points(self):
        return self.points

    def plot(self, ax):
        for point in self.points:
            ax.scatter(point[0], point[1], point[2], color=colors[self.type])


def main():
    # Load dataset
    raw_data = [[int(i) if i.isnumeric() else i for i in line.strip().split()] for line in open("samples/vowdata.dat").readlines()][1:]
    dataset = Dataset(raw_data, 1, 2)
    # Plot feature space
    dataset.plot()


main()
