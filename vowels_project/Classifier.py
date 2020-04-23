import numpy as np
import global_constants
from sklearn.mixture import GaussianMixture


class Classifier:
    def __init__(self, n_components, covariance_type):
        self.gmms = {vowel_type: GaussianMixture(n_components, covariance_type=covariance_type)
                     for vowel_type in
                     global_constants.vowel_types}  # dictionary with a GaussianMixture for every vowel_type

    def train(self, training_set):
        for vowel_type, gmm in self.gmms.items():  # for every vowel_type,
            gmm.fit(training_set[vowel_type])  # fit a gaussian model to the samples

    def classify(self, test_set):
        confusion_matrix = np.zeros((len(global_constants.vowel_types), len(global_constants.vowel_types)), dtype=int)
        vowel_type_to_index = {global_constants.vowel_types[i]: i for i in range(len(global_constants.vowel_types))}
        for vowel_type, samples in test_set.items():
            for sample in samples:  # for all samples
                classify_result = np.argmax([self.gmms[classifier_class].score([sample]) for classifier_class in
                                             global_constants.vowel_types])  # classify the sample by calculating the score for every class and choosing the class with the highest score
                confusion_matrix[classify_result][
                    vowel_type_to_index[vowel_type]] += 1  # increment the appropriate cell in the confusion matrix
        correct_classifications = np.sum([confusion_matrix[i][i] for i in range(len(
            global_constants.vowel_types))])  # sum the diagonal of the confusion matrix to get the number of correct classifications
        return (1 - correct_classifications / np.sum(confusion_matrix)), confusion_matrix
