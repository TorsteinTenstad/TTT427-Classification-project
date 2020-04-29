import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import global_constants


class Classifier:
    def __init__(self, n_components, covariance_type):
        # Dictionary with a GaussianMixture for every vowel_type:
        self.gmms = {vowel_type: GaussianMixture(n_components, covariance_type=covariance_type,
                                                 random_state=0, max_iter=500) for vowel_type in
                     global_constants.vowel_types}

    def train(self, training_set):
        for vowel_type, gmm in self.gmms.items():  # for every vowel_type,
            gmm.fit(training_set[vowel_type])  # fit a gaussian model to the samples

    def classify(self, test_set):
        confusion_matrix = np.zeros(
            (len(global_constants.vowel_types), len(global_constants.vowel_types)), dtype=int)
        vowel_type_to_index = {global_constants.vowel_types[i]: i for i in
                               range(len(global_constants.vowel_types))}
        for vowel_type, samples in test_set.items():
            for sample in samples:  # for all samples
                # Classify the sample by calculating the score for every class
                # and choosing the class with the highest score:
                classify_result = np.argmax(
                    [self.gmms[classifier_class].score([sample]) for classifier_class in
                     global_constants.vowel_types])
                confusion_matrix[classify_result][vowel_type_to_index[
                    vowel_type]] += 1  # increment the appropriate cell in the confusion matrix
        # Sum the diagonal of the confusion matrix to get the number of correct classifications:
        correct_classifications = np.sum(
            [confusion_matrix[i][i] for i in range(len(global_constants.vowel_types))])
        return (1 - correct_classifications / np.sum(confusion_matrix)), confusion_matrix

    def plot_marginal_pdfs(self):
        fig, axs = plt.subplots(3)
        x = np.linspace(0.0, 1.0, num=1000)
        y_max = [15, 11, 8]
        for subplot in range(len(axs)):
            for vowel_type, gmm in self.gmms.items():
                means = gmm.means_[:, subplot]
                standard_deviations = np.sqrt(gmm.covariances_[:, subplot, subplot])
                y = np.zeros(x.shape)
                for i in range(len(gmm.weights_)):
                    y += gmm.weights_[i] * np.exp(
                        -0.5 * np.power((x - means[i]) / standard_deviations[i], 2)) / (
                                 standard_deviations[i] * np.sqrt(2 * np.pi))
                axs[subplot].plot(x, y, c=global_constants.colors[vowel_type], label=vowel_type)
                axs[subplot].set_ylim(0, y_max[subplot])
                axs[subplot].set(xlabel='Normalized F' + str(subplot + 1),
                                 ylabel='Probability density')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=4)
        fig.set_size_inches(4.77, 8.19)
        fig.tight_layout()
        plt.show()
