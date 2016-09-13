import numpy as np

def sigmoid_array(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_gradient_array(x):
	return sigmoid_array(x)*(1-sigmoid_array(x))