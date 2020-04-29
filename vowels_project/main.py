from Dataset import Dataset
from Classifier import Classifier
import confusion_matrix_to_latex_table
import global_constants
import get_raw_data


def main():
    # Load the raw data from dat file:
    raw_data = get_raw_data.from_dat_file('samples/vowdata.dat', duration_mode=2)

    # Normalize the raw data:
    raw_data = get_raw_data.normalize(raw_data)

    # Add a given amount of samples from the raw data into a training set:
    training_set = Dataset(raw_data,
                           samples_to_take_for_each_group={'m': 25, 'w': 25, 'b': 15, 'g': 11})

    # Plot training set:
    training_set.plot()

    # Add the rest of the raw data into a testing set:
    testing_set = Dataset(raw_data)

    # Define the parameters of the classifier variants:
    classifier_variants = [[1, 1, 'diag'], [2, 1, 'full'], [3, 2, 'full'], [4, 3, 'full'],
                           [5, 4, 'full']]

    for variant in classifier_variants:
        # Create a classifier:
        classifier = Classifier(n_components=variant[1], covariance_type=variant[2])

        # Train a classifier based on the training set:
        classifier.train(training_set.data)

        # Classify the testing set with the trained classifier:
        false_classification_rate, confusion_matrix = classifier.classify(testing_set.data)

        # Print results:
        print('Variant ' + str(variant[0]))
        print('False classification rate:', int(100 * false_classification_rate), '%')
        print('Confusion matrix:\n', confusion_matrix, '\n')


main()
