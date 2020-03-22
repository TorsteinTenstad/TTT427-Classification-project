import numpy as np


def get_mean_point(points):
    return np.average(points, axis=0)


def get_covariance_matrix(points):
    return np.cov(np.transpose(points))
