import numpy as np

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
    W = np.zeros((3, len(t1[0])))

    for n in range(num_testruns):
        # Calculating MSE gradient:
        grad_MSE = gradient_MSE(t1, t2, t3, W)
        W = W - alpha * grad_MSE.T

        if n % 2500 == 0:
            print("Training "+ str((n / num_testruns) * 100) + "%")

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