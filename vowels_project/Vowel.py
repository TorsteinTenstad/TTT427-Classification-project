import global_constants
import numpy as np
from scipy.stats import multivariate_normal
import seaborn as sns
import matplotlib.pyplot as plt

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

    def plotGaussian(self, a12, a13, a23, a1, a2, a3):

        f1 = []
        f2 = []
        f3 = []

        for point in self.samples:
            if (point[0] != 0) and (point[1] != 0) and (point[2] != 0):
                f1.append(point[0])
                f2.append(point[1])
                f3.append(point[2])

        sns.kdeplot(f1, f2, kernel='gau', shade=False, shade_lowest=False, n_levels=2, ax=a12, color=global_constants.colors[self.vowel_type])
        sns.kdeplot(f1, f3, kernel='gau', shade=False, shade_lowest=False, n_levels=2, ax=a13, color=global_constants.colors[self.vowel_type])
        sns.kdeplot(f2, f3, kernel='gau', shade=False, shade_lowest=False, n_levels=2, ax=a23, color=global_constants.colors[self.vowel_type])

        sns.scatterplot(f1, f2, ax=a12, color=global_constants.colors[self.vowel_type])
        sns.scatterplot(f1, f3, ax=a13, color=global_constants.colors[self.vowel_type])
        sns.scatterplot(f2, f3, ax=a23, color=global_constants.colors[self.vowel_type])

        sns.kdeplot(f1, kernel='gau', shade=False, ax=a1, color=global_constants.colors[self.vowel_type])
        sns.kdeplot(f2, kernel='gau', shade=False, ax=a2, color=global_constants.colors[self.vowel_type])
        sns.kdeplot(f3, kernel='gau', shade=False, ax=a3, color=global_constants.colors[self.vowel_type])


    def calc_probability(self, point):
        if self.multivariate_normal is None:
            self.calc_multivariate_normal()
        return self.multivariate_normal.pdf(point)
