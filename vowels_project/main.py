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
import pandas as pd


class Vowel:
    def __init__(self, type):
        self.type = type
        self.data = []
        self.mean = 0
        self.covariance = []

    def calc_mean(self):
        self.mean = np.average(points, axis=0)

    def calc_covariance_matrix(self):
        self.covariance = np.cov(np.transpose(self.data))


def main():
    # Load dataset
    dataset = pd.read_fwf("samples/vowdata.dat")
    dataset = dataset.to_numpy()

    print(dataset[:][3])

    # Define list of classes
    vowels = ['ae', 'ah', 'aw', 'ae', 'ah', 'aw', 'eh', 'er', 'ei', 'ih', 'iy', 'oa', 'oo', 'uh', 'uw']
    classes = [None] * 15

    for i in range(len(vowels)):
        classes[i] = Vowel(vowels[i])

    # Add data to classes
    for sample in dataset:
        for ele in classes:
            if sample[0][3:5] == ele.type:
                ele.data.append(sample)

main()
