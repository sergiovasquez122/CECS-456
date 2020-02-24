import numpy as np
from helper import *

'''
Homework2: perceptron classifier
'''


def sign(x):
    return 1 if x > 0 else -1


# -------------- Implement your code Below -------------#

def show_images(data):
    '''
	This function is used for plot image and save it.

	Args:
	data: Two images from train data with shape (2, 16, 16). The shape represents total 2
		  images and each image has size 16 by 16.

	Returns:
		Do not return any arguments, just save the images you plot for your report.
	'''
    image_1 = data[0][:]
    plt.imshow(image_1, cmap='binary')
    plt.axis('off')
    plt.show()

    image_2 = data[1][:]
    plt.imshow(image_2, cmap='binary')
    plt.axis('off')
    plt.show()


def show_features(data, label):
    '''
	This function is used for plot a 2-D scatter plot of the features and save it. 

	Args:
	data: train features with shape (1561, 2). The shape represents total 1561 samples and 
		  each sample has 2 features.
	label: train data's label with shape (1561,1). 
		   1 for digit number 1 and -1 for digit number 5.

	Returns:
	Do not return any arguments, just save the 2-D scatter plot of the features you plot for your report.
	'''

    colors = ('blue', 'red')
    markers = ('+', 'o')
    for idx, cl in enumerate(np.unique(label)):
        plt.scatter(data[cl == label, 0], data[cl == label, 1], color=colors[idx], marker=markers[idx],
                    label=cl, edgecolors='black')
    plt.legend(loc='upper left')
    plt.xlabel('Symmetry')
    plt.ylabel('Average Intensity')
    plt.show()


def perceptron(data, label, max_iter, learning_rate):
    '''
	The perceptron classifier function.

	Args:
	data: train data with shape (1561, 3), which means 1561 samples and 
		  each sample has 3 features.(1, symmetry, average internsity)
	label: train data's label with shape (1561,1). 
		   1 for digit number 1 and -1 for digit number 5.
	max_iter: max iteration numbers
	learning_rate: learning rate for weight update

	Returns:
		w: the seperater with shape (1, 3). You must initilize it with w = np.zeros((1,d))
	'''
    w = np.zeros((1, 3))
    for iteration in range(max_iter):
        for xi, yi in zip(data, label):
            prediction = xi.dot(w.T)
            if sign(prediction) != yi:
                w = w + learning_rate * xi * yi
    return w


def show_result(data, label, w):
    '''
	This function is used for plot the test data with the separators and save it.

	Args:
	data: test features with shape (424, 2). The shape represents total 424 samples and 
		  each sample has 2 features.
	label: test data's label with shape (424,1). 
		   1 for digit number 1 and -1 for digit number 5.

	Returns:
	Do not return any arguments, just save the image you plot for your report.

	'''

    x1_min, x1_max = data[0, :].min() - 0.5, data[0, :].max() + 0.5
    x2_min, x2_max = data[1, :].min() - 0.5, data[1, :].max() + 0.5
    x1_values = np.linspace(x1_min, x1_max, 100)
    x2_values = np.asarray([-((w[0][0] + w[0][1] * xi)/w[0][2]) for xi in x1_values])

    colors = ('blue', 'red')
    markers = ('+', 'o')
    for idx, cl in enumerate(np.unique(label)):
        plt.scatter(data[cl == label, 0], data[cl == label, 1], color=colors[idx], marker=markers[idx],
                    label=cl, edgecolors='black')
    plt.legend(loc='upper left')
    plt.xlabel('Symmetry')
    plt.ylabel('Average Intensity')
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.plot(x1_values, x2_values, color = 'black')
    plt.show()


# -------------- Implement your code above ------------#
def accuracy_perceptron(data, label, w):
    n, _ = data.shape
    mistakes = 0
    for i in range(n):
        if sign(np.dot(data[i, :], np.transpose(w))) != label[i]:
            mistakes += 1
    return (n - mistakes) / n


def test_perceptron(max_iter, learning_rate):
    # get data
    traindataloc, testdataloc = "../data/train.txt", "../data/test.txt"
    train_data, train_label = load_features(traindataloc)
    test_data, test_label = load_features(testdataloc)
    # train perceptron
    w = perceptron(train_data, train_label, max_iter, learning_rate)
    train_acc = accuracy_perceptron(train_data, train_label, w)
    # test perceptron model
    test_acc = accuracy_perceptron(test_data, test_label, w)
    return w, train_acc, test_acc
