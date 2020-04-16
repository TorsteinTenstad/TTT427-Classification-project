import global_constants
import numpy as np
from sklearn.mixture import GaussianMixture


class Vowel:
    def __init__(self, vowel_type, n_components=1, covariance_type='diag'):
        self.vowel_type = vowel_type
        self.samples = []
        self.gmm = GaussianMixture(n_components, covariance_type=covariance_type)

    def fit_gmm(self):
        self.gmm.fit(self.samples)

    def score(self, point):
        return self.gmm.score([point])

    def add_point(self, point):
        self.samples.append(np.array(point))

    def get_samples(self):
        return self.samples

    def get_type(self):
        return self.vowel_type

    def plot(self, ax):
        for point in self.samples:
            ax.scatter(point[0], point[1], point[2], color=global_constants.colors[self.vowel_type])
