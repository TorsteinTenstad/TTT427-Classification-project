import numpy as np
from matplotlib import pyplot as plt
from test_and_train_classifier import *
from plot_histogram import plot_histogram

plt.style.use('seaborn-whitegrid')

traning_length = 10000

def task1():
    # Load data from csv-file
    data_class1 = np.loadtxt(open("Data/class_1", "rb"), delimiter=",", skiprows=1)
    data_class2 = np.loadtxt(open("Data/class_2", "rb"), delimiter=",", skiprows=1)
    data_class3 = np.loadtxt(open("Data/class_3", "rb"), delimiter=",", skiprows=1)

    print("Part A: First 30 samples for training, last 20 for testing")

    # Find weight matrix with 30 first samples as training set
    W = train_weight_matrix(traning_length, 0.01, data_class1[:30, :], data_class2[:30, :], data_class3[:30, :])
    print("Weight matrix")
    print(W)

    # Find confusion matrix, for test set
    print("Confusion matrix for testing set")
    confusion_matrix_test, err_test = test_classifier(W, data_class1[30:50, :], data_class2[30:50, :], data_class3[30:50, :])
    print(confusion_matrix_test)
    print("Error rate: " + "%.2f" % (err_test*100) + "%")

    print("Confusion matrix for training set")
    confusion_matrix_training, err_train = test_classifier(W, data_class1[:30, :], data_class2[:30, :], data_class3[:30, :])
    print(confusion_matrix_training)
    print("Error rate: " + "%.2f" % (err_train*100) + "%")

    print("Total confusion matrix")
    print(confusion_matrix_test + confusion_matrix_training)

    # ==================================================================================================================

    print("Part B: Last 30 samples for training, first 20 for testing")

    # Find weight matrix with 30 first samples as training set
    W = train_weight_matrix(traning_length, 0.01, data_class1[20:50, :], data_class2[20:50, :], data_class3[20:50, :])
    print("Weight matrix")
    print(W)

    # Find confusion matrix, for test set
    print("Confusion matrix for testing set")
    confusion_matrix_test, err_test = test_classifier(W, data_class1[:20, :], data_class2[:20, :], data_class3[:20, :])
    print(confusion_matrix_test)
    print("Error rate: " + "%.2f" % (err_test*100) + "%")

    print("Confusion matrix for training set")
    confusion_matrix_training, err_train = test_classifier(W, data_class1[20:50, :], data_class2[20:50, :], data_class3[20:50, :])
    print(confusion_matrix_training)
    print("Error rate: " + "%.2f" % (err_train*100) + "%")

    print("Total confusion matrix")
    print(confusion_matrix_test + confusion_matrix_training)

def task2():
    # Plot histogram
    plot_histogram()

    # Load data and split into training and testing set
    data_class1 = np.loadtxt(open("Data/class_1", "rb"), delimiter=",", skiprows=1)
    data_class2 = np.loadtxt(open("Data/class_2", "rb"), delimiter=",", skiprows=1)
    data_class3 = np.loadtxt(open("Data/class_3", "rb"), delimiter=",", skiprows=1)

    training1 = data_class1[:30, :]
    training2 = data_class2[:30, :]
    training3 = data_class3[:30, :]

    testing1 = data_class1[30:50, :]
    testing2 = data_class2[30:50, :]
    testing3 = data_class3[30:50, :]

    # Train and test without sepal width
    testing1 = np.delete(testing1, 1, 1)
    testing2 = np.delete(testing2, 1, 1)
    testing3 = np.delete(testing3, 1, 1)

    training1 = np.delete(training1, 1, 1)
    training2 = np.delete(training2, 1, 1)
    training3 = np.delete(training3, 1, 1)

    W = train_weight_matrix(traning_length, 0.01, training1, training2, training3)
    confusion_matrix, err = test_classifier(W, testing1, testing2, testing3)
    print("Without sepal width")
    print(confusion_matrix)
    print("Error rate: " + "%.2f" % (err * 100) + "%")

    # Train and test without sepal length
    testing1 = np.delete(testing1, 0, 1)
    testing2 = np.delete(testing2, 0, 1)
    testing3 = np.delete(testing3, 0, 1)

    training1 = np.delete(training1, 0, 1)
    training2 = np.delete(training2, 0, 1)
    training3 = np.delete(training3, 0, 1)

    W = train_weight_matrix(traning_length, 0.01, training1, training2, training3)
    confusion_matrix, err = test_classifier(W, testing1, testing2, testing3)
    print("Without sepal length and sepal width")
    print(confusion_matrix)
    print("Error rate: " + "%.2f" % (err * 100) + "%")

    # Train and test without petal width
    testing1 = np.delete(testing1, 1, 1)
    testing2 = np.delete(testing2, 1, 1)
    testing3 = np.delete(testing3, 1, 1)

    training1 = np.delete(training1, 1, 1)
    training2 = np.delete(training2, 1, 1)
    training3 = np.delete(training3, 1, 1)

    W = train_weight_matrix(traning_length, 0.01, training1, training2, training3)
    confusion_matrix, err = test_classifier(W, testing1, testing2, testing3)
    print("Without sepal length, sepal width and petal width")
    print(confusion_matrix)
    print("Error rate: " + "%.2f" % (err * 100) + "%")

def main():
    print("======  task1  ======")
    task1()

    print("======  task2  ======")
    task2()


main()
