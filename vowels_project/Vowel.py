import global_constants
import numpy as np


class Vowel:
    def __init__(self, vowel_type):
        self.type = vowel_type
        self.points = []
        self.mean = 0
        self.covariance = []

    def calc_mean(self):
        self.mean = np.average(self.points, axis=0)

    def calc_covariance_matrix(self):
        self.covariance = np.cov(np.transpose(self.points))

    def add_point(self, point):
        self.points.append(point)

    def get_points(self):
        return self.points

    def plot(self, ax):
        for point in self.points:
            ax.scatter(point[0], point[1], point[2], color=global_constants.colors[self.type])
