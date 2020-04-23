from Dataset import Dataset
from Classifier import Classifier
import confusion_matrix_to_latex_table
import global_constants
import get_raw_data


def main():
    # Load the raw data from dat file:
    raw_data = get_raw_data.from_dat_file('samples/vowdata.dat', mode=2)

    # Alternatively, one can load raw data using analysis of wav files, but this is slow, and gives worse performance:
    # raw_data = get_raw_data.from_wav_analysis('samples')

    # Normalize the raw data:
    raw_data = get_raw_data.normalize(raw_data)

    # Add a given amount of samples from the raw data into a training set:
    training_set = Dataset(raw_data, samples_to_take_for_each_group={'m': 25, 'w': 25, 'b': 15, 'g': 11})

    # Add the rest of the raw data into a testing set:
    testing_set = Dataset(raw_data)

    # Create a classifier:
    classifier = Classifier(n_components=3, covariance_type='full')

    # Train a classifier based on the training set:
    classifier.train(training_set.data)

    # Classify the testing set with the trained classifier:
    false_classification_rate, confusion_matrix = classifier.classify(testing_set.data)

    # Print results:
    print('False classification rate:', int(100 * false_classification_rate), '%')
    print('Confusion matrix:\n', confusion_matrix)

    # Format confusion matrix for latex:
    confusion_matrix_to_latex_table.to_latex_table(confusion_matrix, global_constants.vowel_types, 'confusion_matrix.txt')


main()
