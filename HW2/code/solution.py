import numpy as np 
from helper import *
'''
Homework2: logistic regression classifier
'''


def sigmoid(z):
    '''
    Computes the sigmoid activation function

    :param z: numpy array
    :return: The sigmoid value of z
    '''
    return 1 / (1 + np.exp(-z))

def logistic_regression(data, label, max_iter, learning_rate):
    '''
    The logistic regression classifier function.

    Args:
    data: train data with shape (1561, 3), which means 1561 samples and
          each sample has 3 features.(1, symmetry, average internsity)
    label: train data's label with shape (1561,1).
           1 for digit number 1 and -1 for digit number 5.
    max_iter: max iteration numbers
    learning_rate: learning rate for weight update

    Returns:
        w: the seperater with shape (3, 1). You must initilize it with w = np.zeros((d,1))
    '''
    w = np.zeros((data.shape[1], 1))
    for current_epoch in range(max_iter):
        v_t = np.zeros((data.shape[1],1))
        for (xi, yi) in zip(data, label):
            xi = xi.reshape((data.shape[1],1))
            v_t += yi * xi * sigmoid(-yi * w.T.dot(xi))
        v_t /= len(label)
        w = w + v_t * learning_rate
    return w

def thirdorder(data):
    '''
    This function is used for a 3rd order polynomial transform of the data.
    Args:
    data: input data with shape (:, 3) the first dimension represents
          total samples (training: 1561; testing: 424) and the
          second dimesion represents total features.

    Return:
        result: A numpy array format new data with shape (:,10), which using
        a 3rd order polynomial transformation to extend the feature numbers
        from 3 to 10.
        The first dimension represents total samples (training: 1561; testing: 424)
        and the second dimesion represents total features.
    '''
    degree = 3
    output = np.ones((data.shape[0], 1))
    x1 = data[:, 0]
    x2 = data[:, 1]
    for i in range(1, degree + 1):
        for j in range(0, i + 1):
            col = (x1 ** (i - j)) * (x2 ** j)
            col = col.reshape(col.shape[0], 1)
            output = np.append(output, col, axis = 1)
    return output


def accuracy(x, y, w):
    '''
    This function is used to compute accuracy of a logsitic regression model.
    
    Args:
    x: input data with shape (n, d), where n represents total data samples and d represents
        total feature numbers of a certain data sample.
    y: corresponding label of x with shape(n, 1), where n represents total data samples.
    w: the seperator learnt from logistic regression function with shape (d, 1),
        where d represents total feature numbers of a certain data sample.

    Return 
        accuracy: total percents of correctly classified samples. Set the threshold as 0.5,
        which means, if the predicted probability > 0.5, classify as 1; Otherwise, classify as -1.
    '''
    activation = sigmoid(np.dot(x,w))
    predictions = np.where(activation > 0.5, 1, -1)
    y = y.reshape((y.shape[0], 1))
    return np.mean(predictions == y)

