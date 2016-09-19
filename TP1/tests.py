import numpy as np

import preprocessor
import sigmoid
import ej1_data_loader
from layer_model import LayerModel
from feed_forward_solver import FeedForwardSolver
import functools

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

#print data

features = data[0] #shape=(333,10)
labels = data[1]
#print ("labels shape ", labels.shape)
#labels.shape = (1,410)

T = 200
t = 0
epsilon = 0.01
e = 999
num_features=features.shape[1]
model = LayerModel(num_features , [num_features + 2], 1, functools.partial( sigmoid.sigmoid_array,1),functools.partial( sigmoid.sigmoid_gradient_array,1))
W = model.getInitializedWeightMats()
print(W)


ffsolver = FeedForwardSolver(W,model)

while e > epsilon and t < T:
    e = ffsolver.batch(features, labels)
    print ("Error: " , e)
    t = t + 1