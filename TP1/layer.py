import functools

import sigmoid


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

