import numpy as np
from matplotlib import pyplot as plt

plt.style.use('seaborn-whitegrid')

def sigmoid(x, W):
    return 1 / (1 + np.e ** (-np.matmul(W, x)))


def gradient_MSE(t1, t2, t3, W):
    MSE = 0

    t = np.transpose([1, 0, 0])
    for x in t1:
        g = sigmoid(x, W)

        x = np.array([x])

        MSE = MSE + ((g - t).dot(g.dot(1 - g))) * x.T

    t = np.transpose([0, 1, 0])
    for x in t2:
        g = sigmoid(x, W)

        x = np.array([x])

        MSE = MSE + ((g - t).dot(g.dot(1 - g))) * x.T

    t = np.transpose([0, 0, 1])
    for x in t3:
        g = sigmoid(x, W)

        x = np.array([x])

        MSE = MSE + ((g - t).dot(g.dot(1 - g))) * x.T

    return MSE


def train_weight_matrix(num_testruns, alpha, t1, t2, t3):
    W = np.zeros((3, 4))

    for n in range(num_testruns):
        # Calculating MSE gradient:
        grad_MSE = gradient_MSE(t1, t2, t3, W)
        W = W - alpha * grad_MSE.T

        if n % 2500 == 0:
            print(str((n / num_testruns) * 100) + "%")

    return W


def test_classifier(W, t1, t2, t3):
    confusion_matrix = np.zeros((3, 3))

    for x in t1:
        g = np.matmul(W, x)
        i = np.argmax(g)
        confusion_matrix[0][i] = confusion_matrix[0][i] + 1

    for x in t2:
        g = np.matmul(W, x)
        i = np.argmax(g)
        confusion_matrix[1][i] = confusion_matrix[1][i] + 1

    for x in t3:
        g = np.matmul(W, x)
        i = np.argmax(g)
        confusion_matrix[2][i] = confusion_matrix[2][i] + 1

    err = 1- (confusion_matrix[0][0] + confusion_matrix[1][1] + confusion_matrix[2][2])/np.sum(confusion_matrix)


    return confusion_matrix, err


def task1():
    # Load data from csv-file
    data_class1 = np.loadtxt(open("Data/class_1", "rb"), delimiter=",", skiprows=1)
    data_class2 = np.loadtxt(open("Data/class_2", "rb"), delimiter=",", skiprows=1)
    data_class3 = np.loadtxt(open("Data/class_3", "rb"), delimiter=",", skiprows=1)

    print("Part A: First 30 samples for training, last 20 for testing")

    # Find weight matrix with 30 first samples as training set
    W = train_weight_matrix(10000, 0.01, data_class1[:30, :], data_class2[:30, :], data_class3[:30, :])
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
    W = train_weight_matrix(10000, 0.01, data_class1[20:50, :], data_class2[20:50, :], data_class3[20:50, :])
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
    # Load data from csv-file
    data_class1 = np.loadtxt(open("Data/class_1", "rb"), delimiter=",", skiprows=1)
    data_class2 = np.loadtxt(open("Data/class_2", "rb"), delimiter=",", skiprows=1)
    data_class3 = np.loadtxt(open("Data/class_3", "rb"), delimiter=",", skiprows=1)

    # Plot histogram
    fig, axs = plt.subplots(3, 4, sharey=True, sharex='col', tight_layout=True)
    n_bins = 15

    #Class 1
    axs[0,0].hist(data_class1[:, 0], bins = n_bins)
    axs[0,1].hist(data_class1[:, 1], bins = n_bins)
    axs[0,2].hist(data_class1[:, 2], bins = n_bins)
    axs[0,3].hist(data_class1[:, 3], bins = n_bins)

    #Class 2
    axs[1, 0].hist(data_class2[:, 0], bins = n_bins)
    axs[1, 1].hist(data_class2[:, 1], bins = n_bins)
    axs[1, 2].hist(data_class2[:, 2], bins = n_bins)
    axs[1, 3].hist(data_class2[:, 3], bins = n_bins)

    # Class 3
    axs[2, 0].hist(data_class3[:, 0], bins = n_bins)
    axs[2, 1].hist(data_class3[:, 1], bins = n_bins)
    axs[2, 2].hist(data_class3[:, 2], bins = n_bins)
    axs[2, 3].hist(data_class3[:, 3], bins = n_bins)

    axs[0, 0].set_ylabel('Class 1')
    axs[1, 0].set_ylabel('Class 2')
    axs[2, 0].set_ylabel('Class 3')

    axs[0, 0].set_title('Sepal length')
    axs[0, 1].set_title('Sepal width')
    axs[0, 2].set_title('Petal length')
    axs[0, 3].set_title('Petal width')

    plt.show()


def main():
    print("======  task1  ======")
    #task1()

    print("======  task2  ======")
    task2()


main()
