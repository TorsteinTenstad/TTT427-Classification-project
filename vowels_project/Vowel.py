import global_constants
import numpy as np
from scipy.stats import multivariate_normal


class Vowel:
    def __init__(self, vowel_type):
        self.vowel_type = vowel_type
        self.samples = []
        self.mean = []
        self.covariance = []
        self.multivariate_normal = None

    def calc_mean(self):
        self.mean = np.average(self.samples, axis=0)

    def calc_covariance_matrix(self):
        self.covariance = np.cov(np.transpose(self.samples))

    def calc_multivariate_normal(self):
        if len(self.mean) == 0:
            self.calc_mean()
        if len(self.covariance) == 0:
            self.calc_covariance_matrix()
        self.multivariate_normal = multivariate_normal(mean=self.mean, cov=self.covariance)

    def make_covariance_matrix_diagonal(self):
        n = len(self.covariance)
        for i in range(n):
            for j in range(n):
                if i != j:
                    self.covariance[i][j] = 0

    def add_point(self, point):
        self.samples.append(point)

    def get_samples(self):
        return self.samples

    def get_type(self):
        return self.vowel_type

    def plot(self, ax):
        for point in self.samples:
            ax.scatter(point[0], point[1], point[2], color=global_constants.colors[self.vowel_type])

    def calc_probability(self, point):
        if self.multivariate_normal is None:
            self.calc_multivariate_normal()
        return self.multivariate_normal.pdf(point)
