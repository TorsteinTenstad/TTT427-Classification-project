import numpy as np
from matplotlib import pyplot as plt

def plot_histogram():
    # Load data from csv-file
    data_class1 = np.loadtxt(open("Data/class_1", "rb"), delimiter=",", skiprows=1)
    data_class2 = np.loadtxt(open("Data/class_2", "rb"), delimiter=",", skiprows=1)
    data_class3 = np.loadtxt(open("Data/class_3", "rb"), delimiter=",", skiprows=1)

    # Plot histogram
    fig, axs = plt.subplots(3, 4, sharey=True, sharex='col', tight_layout=True)

    #Class 1
    axs[0,0].hist(data_class1[:, 0])
    axs[0,1].hist(data_class1[:, 1])
    axs[0,2].hist(data_class1[:, 2])
    axs[0,3].hist(data_class1[:, 3])

    #Class 2
    axs[1, 0].hist(data_class2[:, 0])
    axs[1, 1].hist(data_class2[:, 1])
    axs[1, 2].hist(data_class2[:, 2])
    axs[1, 3].hist(data_class2[:, 3])

    # Class 3
    axs[2, 0].hist(data_class3[:, 0])
    axs[2, 1].hist(data_class3[:, 1])
    axs[2, 2].hist(data_class3[:, 2])
    axs[2, 3].hist(data_class3[:, 3])

    axs[0, 0].set_ylabel('Class 1')
    axs[1, 0].set_ylabel('Class 2')
    axs[2, 0].set_ylabel('Class 3')

    axs[0, 0].set_title('Sepal length')
    axs[0, 1].set_title('Sepal width')
    axs[0, 2].set_title('Petal length')
    axs[0, 3].set_title('Petal width')

    plt.show()