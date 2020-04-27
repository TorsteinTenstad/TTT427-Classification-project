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
from Dataset import Dataset
import sound_processing


def get_raw_data_from_dat_file(file):
    return [[int(i) if i.isnumeric() else i for i in line.strip().split()] for line in open(file).readlines()][1:]


def get_raw_data_from_wav_analysis(folder):
    raw_data = []
    for subfolder in ['men', 'women', 'kids']:
        raw_data = raw_data + sound_processing.get_raw_feature_data(folder + '/' + subfolder)
    return raw_data


def main():
    # Load the raw data:
    raw_data = get_raw_data_from_dat_file('samples/vowdata.dat')
    # raw_data = get_raw_data_from_wav_analysis('samples')

    # Add a given amount of samples from the raw data into a training set:
    training_set = Dataset(raw_data, samples_to_take_for_each_group={'m': 25, 'w': 25, 'b': 15, 'g': 11})

    # Add the rest of the raw data into a testing set:
    testing_set = Dataset(raw_data)

    # Train a classifier based on the training set:
    training_set.train()

    # Classify the testing set with the trained classifier:
    false_classification_rate, confusion_matrix = training_set.classify_test_set(testing_set)

    # Print results:
    print('False classification rate:', int(100 * false_classification_rate), '%')

    # Constrain the classifier to use diagonal covariance matrices:
    training_set.make_all_covariance_matrices_diagonal()

    # Train the classifier with this constraint:
    training_set.train()

    # Classify the testing set with the new classifier:
    false_classification_rate, confusion_matrix = training_set.classify_test_set(testing_set)

    # Print results:
    print('False classification rate with diagonal covariance matrices:', int(100 * false_classification_rate), '%')

    training_set.plot_gaussian()

main()
