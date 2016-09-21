import numpy as np

import preprocessor
import sigmoid
import ej1_data_loader
from layer_model import LayerModel
from feed_forward_solver import NetworkSolver
import functools

loader = ej1_data_loader.Ej1DataLoader()
data = loader.LoadData()
features = data[0] #shape=(333,10)
labels = data[1]

training_data = zip(features, labels)
mini_batch_size = 1
n = len(training_data)
mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]

#mini_batch = zip(features, labels)

model =  LayerModel([10,12,1], functools.partial( sigmoid.sigmoid_array,1),functools.partial( sigmoid.sigmoid_gradient_array,1))
solver = NetworkSolver(layer_model=model)

solver.learn_minibatch(mini_batches,0.005,900,0.01)
