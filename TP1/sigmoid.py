import numpy as np


def sigmoid_array(b, x):
    return 1.0 / (1.0 + np.exp(-b * x))


def sigmoid_gradient_array(b, x):
    return b * sigmoid_array(b, x) * (1.0 - sigmoid_array(b, x))


def sigmoid_tanh_array(b, x):
    return np.tanh(b * x)


def sigmoid_tanh_gradient_array(b, x):
    return b * (1 - (np.power(sigmoid_tanh_array(b, x), 2)))
