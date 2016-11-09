import functools

import sigmoid

import numpy as np


class Layer:
    def get_num_layers(self):
        return self._num_layers

    def activation(self, z):
        raise NotImplementedError('subclass responsibility')

    def derivative(self, z):
        raise NotImplementedError('subclass responsibility')

class SigmoidLayer(Layer):
    def __init__(self, num_layers, beta):
        self._num_layers = num_layers
        self._activation = functools.partial( sigmoid.sigmoid_array,beta)
        self._derivative =  functools.partial( sigmoid.sigmoid_gradient_array,beta)

    def activation(self, z):
        return self._activation(z)

    def derivative(self, z):
        return self._derivative(z)

class InputLayer(Layer):
    def __init__(self, num_layers):
        self._num_layers = num_layers

    def activation(self, z):
        raise NotImplementedError('input layers do not activate')

    def derivative(self, z):
        raise NotImplementedError('input layers do not define derivative')

class ReluLayer(Layer):
    def __init__(self, num_layers, beta):
        self._num_layers = num_layers
        self._activation = lambda z: z / beta if z > 0 else 0
        self._derivative =  lambda z: 1.0 / beta # Esto no esta bien alrededor de 0

    def activation(self, z):
        return self._activation(z)

    def derivative(self, z):
        return self._derivative(z)

class LinearLayer(Layer):
    def __init__(self, num_layers, beta):
        self._num_layers = num_layers
        self._activation = lambda z: z / beta
        self._derivative = lambda z: 1.0 / beta

    def activation(self, z):
        return self._activation(z)

    def derivative(self, z):
        return self._derivative(z)

class TanhLayer(Layer ):
    def __init__(self, num_layers, beta):
        self._num_layers = num_layers
        self._activation = lambda z: np.tanh(z / beta)
        self._derivative = lambda z: (1.0 - np.tanh(z/beta)**2) / beta

    def activation(self, z):
        return self._activation(z)

    def derivative(self, z):
        return self._derivative(z)