import numpy as np

import preprocessor
import sigmoid
import ej1_data_loader
from feed_forward_solver import FeedForwardSolver

# X = np.genfromtxt('/Users/bpanarello/Dropbox/RN/P2/letters_1.txt', delimiter=" ")
# B = np.ones((26, 1)) * -1
# X = np.concatenate((B, X), axis=1)
# Z = np.genfromtxt('/Users/bpanarello/Dropbox/RN/P2/letters_tags.txt', delimiter=" ")
# W = np.random.rand(26, 5)/100


# solver = FeedForwardSolver([W], sigmoid.sigmoid_array)
# Y = solver.solve_sample_and_return_output(X)
# print Y


X = np.genfromtxt('./ds/tp1_ej1_training.csv', delimiter=",")
loader = ej1_data_loader.Ej1DataLoader()
loader.LoadData()
print N

