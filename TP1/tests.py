import numpy as np

import preprocessor
import sigmoid
import ej1_data_loader
from layer_model import LayerModel
from feed_forward_solver import FeedForwardSolver

# X = np.genfromtxt('/Users/bpanarello/Dropbox/RN/P2/letters_1.txt', delimiter=" ")
# B = np.ones((26, 1)) * -1
# X = np.concatenate((B, X), axis=1)
# Z = np.genfromtxt('/Users/bpanarello/Dropbox/RN/P2/letters_tags.txt', delimiter=" ")
# W = np.random.rand(26, 5)/100


# solver = FeedForwardSolver([W], sigmoid.sigmoid_array)
# Y = solver.solve_sample_and_return_output(X)
# print Y

loader = ej1_data_loader.Ej1DataLoader()
data = loader.LoadData()

print data

features = data[0]
labels = data[1]

epochs = 40
t = 0
epsilon = 0.01
e = 999
model = LayerModel(len(features) + 1, [len(features) + 2], 1, sigmoid.sigmoid_array)
W = model.getInitializedWeightMats()
ffsolver = FeedForwardSolver(W, sigmoid.sigmoid_array)
E = np.array(1)
coef = 0.01
while (t < epochs) and (e < epsilon):
    e = 0
    i = 0
    for j in range(0, len(data)-1):
        Y = ffsolver.solve_sample_and_return_output(features[j], W)
        E[j] = labels[j] - Y
        dw = coef * np.inner(np.transpose(X), E)
        W += dw
        i += 1
        e += np.linalg.norm(E) ** 2
t += 1








