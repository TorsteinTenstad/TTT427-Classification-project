import global_constants
import numpy as np
from scipy.stats import multivariate_normal


class Vowel:
    def __init__(self, vowel_type):
        self.type = vowel_type
        self.samples = []
        self.mean = []
        self.covariance = []

    def calc_mean(self):
        self.mean = np.average(self.samples, axis=0)

    def calc_covariance_matrix(self):
        self.covariance = np.cov(np.transpose(self.samples))

    def add_point(self, point):
        self.samples.append(point)

    def get_samples(self):
        return self.samples

    def plot(self, ax):
        for point in self.samples:
            ax.scatter(point[0], point[1], point[2], color=global_constants.colors[self.type])

    def calc_probability(self, point):
        if len(self.mean) == 0:
            self.calc_mean()
        if len(self.covariance) == 0:
            self.calc_covariance_matrix()
        print(self.mean)
        print(self.covariance)
        return multivariate_normal(mean=self.mean, cov=self.covariance).pdf(point)
