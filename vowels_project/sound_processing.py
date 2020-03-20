import numpy as np
import scipy.io.wavfile
import scipy.signal
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join

colors = {"ae": "black",
          "ah": "dimgrey",
          "aw": "silver",
          "eh": "lightcoral",
          "ei": "firebrick",
          "er": "crimson",
          "ih": "darkorange",
          "iy": "yellow",
          "oa": "lime",
          "oo": "darkgreen",
          "uh": "cornflowerblue",
          "uw": "mediumblue"}



def get_sound_signal(filename):
    fs, data = scipy.io.wavfile.read(filename)
    return np.array(data, dtype=float)


def fit_AR_model(signal, order):
    autocorrelation = np.correlate(signal, signal, mode='full')
    middle_index = int(len(autocorrelation - 1) / 2)
    Gamma_xx = np.empty([order, order])
    for i in range(order):
        Gamma_xx[i] = autocorrelation[middle_index - i:middle_index + order - i]
    model_coefficients = np.dot(-np.linalg.inv(Gamma_xx), autocorrelation[middle_index + 1:middle_index + 1 + order])
    return model_coefficients


def get_PSD_of_AR_model(model_coefficients, resolution):
    f = np.arange(resolution) / (2*resolution)
    model_order = len(model_coefficients)
    A = np.zeros(resolution, dtype=np.dtype('complex128'))
    for i in range(model_order):
        A += model_coefficients[i]*np.exp(-1j*2*np.pi*f*(-(i+1)))
    PSD = 1/(np.square(abs(1+A)))
    return PSD


def get_point_in_feature_space(filename, plot=False, feature_space_dimension=3, min_distance_between_frequency_peaks=0.025, order=15):
    signal = get_sound_signal(filename)
    a = fit_AR_model(signal, order)
    resolution = int((len(signal)/2))
    f = np.arange(resolution) / (2*resolution)
    PSD = get_PSD_of_AR_model(a, resolution)
    peak_index, properties = scipy.signal.find_peaks(PSD, distance=min_distance_between_frequency_peaks*2*resolution, height=1)
    peak_heights = properties['peak_heights']
    if len(peak_index) < feature_space_dimension:
        return get_point_in_feature_space(filename, plot, feature_space_dimension, min_distance_between_frequency_peaks, order=order+1)
    elif len(peak_index > feature_space_dimension):
        peak_index = peak_index[0:feature_space_dimension]
        peak_heights = peak_heights[0:feature_space_dimension]
    frequency_peaks = f[peak_index]
    if plot:
        signal_fft = abs(np.fft.fft(signal))[0:resolution]
        plt.plot(f, signal_fft/max(signal_fft))
        plt.plot(f, PSD/max(PSD))
        plt.plot(frequency_peaks, peak_heights/max(peak_heights), 'ro')
        plt.show()
    return frequency_peaks


def get_points_in_feature_space(folder_with_sound_samples):
    files = [f for f in listdir(folder_with_sound_samples) if isfile(join(folder_with_sound_samples, f))]
    points = {}
    for file in files:
        point = get_point_in_feature_space(folder_with_sound_samples+'/'+file)
        vowel = file[3:5]
        if vowel in points.keys():
            points[vowel].append(point)
        else:
            points[vowel] = [point]
    return points


def plot_points(points_dict):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for vowel, points in points_dict.items():
        for point in points:
            ax.scatter(point[0], point[1], point[2], color=colors[vowel])
    plt.show()
