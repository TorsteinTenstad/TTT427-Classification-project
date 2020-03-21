import numpy as np

def sigmoid(x,W):
    return 1/(1+np.e**(-np.matmul(W,x)))

def gradient_MSE(t1, t2, t3, W):
    MSE = 0

    t = np.transpose([1, 0, 0])
    for x in t1:
        g = sigmoid(x, W)

        x = np.array([x])

        MSE = MSE + ((g - t).dot(g.dot(1 - g)))*x.T


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
            print(str((n/num_testruns)*100) + "%")

    return W

def test_classifier(W, t1,t2,t3):

    confusion_matrix = np.zeros((2, 3))

    for x in t1:

        g = np.matmul(W,x)

        if g[0] > g[1] and g[0] > g[2]:
            confusion_matrix[0][0] = confusion_matrix[0][0] + 1

        else:
            confusion_matrix[1][0] = confusion_matrix[1][0] + 1

    for x in t2:

        g = np.matmul(W,x)

        if g[1] > g[0] and g[1] > g[2]:
            confusion_matrix[0][1] = confusion_matrix[0][1] + 1

        else:
            confusion_matrix[1][1] = confusion_matrix[1][1] + 1


    for x in t3:

        g = np.matmul(W,x)

        if g[2] > g[0] and g[2] > g[1]:
            confusion_matrix[0][2] = confusion_matrix[0][2] + 1

        else:
            confusion_matrix[1][2] = confusion_matrix[1][2] + 1

    return confusion_matrix

def task1():

    #====== Load data ======

    data_class1 = np.loadtxt(open("Data/class_1", "rb"), delimiter=",", skiprows=1)
    data_class2 = np.loadtxt(open("Data/class_2", "rb"), delimiter=",", skiprows=1)
    data_class3 = np.loadtxt(open("Data/class_3", "rb"), delimiter=",", skiprows=1)

    #====== Define set for training and testing ======
    # First 30 as training set, last 20 as verification

    train_class1 = data_class1[:30, :]
    test_class1 = data_class1[30:50, :]

    train_class2 = data_class2[:30, :]
    test_class2 = data_class2[30:50, :]

    train_class3 = data_class3[:30, :]
    test_class3 = data_class3[30:50, :]

    #====== Training ======
    print("Trener!")
    W = train_weight_matrix(1000000, 0.01, train_class1, train_class2, train_class3)

    #====== Testing =======
    print("Tester!")
    print("testverdier")
    confusion_matrix = test_classifier(W, test_class1, test_class2, test_class3)
    print(confusion_matrix)
    print("treningsverdier")
    confusion_matrix = test_classifier(W, train_class1, train_class2, train_class3)
    print(confusion_matrix)


def main():
    print("======  task1  ======")
    task1()

main()