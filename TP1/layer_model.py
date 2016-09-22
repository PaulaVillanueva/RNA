import numpy as np
class LayerModel:

    # numInputUnits: Cantidad de unidades de entrada
    # hiddenLayers: Array con la cantidad de hidden layers de la red
    # numOutputLayers: Cantidad de unidades de salida
    def __init__(self, layers):
        self._layers = layers
        self._layer_sizes = [l.get_num_layers() for l in layers]
        self._activations_fns = [l.activation for l in layers]
        self._derivative_fns = [l.derivative for l in layers]

        self._biases = [np.random.randn(y, 1) / 1000 for y in self._layer_sizes[1:]]
        self._weights = [np.random.randn(y, x) / 1000
                        for x, y in zip(self._layer_sizes[:-1], self._layer_sizes[1:])]
        self._num_layers = len(self._layer_sizes)

    def getInitializedWeightMats(self):
        return self._weights

    def getInitializedBiasVectors(self):
        return self._biases

    def getZeroDeltaW(self):
        return [np.zeros(w.shape) for w in self._weights]

    def getZeroDeltaB(self):
        return [np.zeros(b.shape) for b in self._biases]

    def activation(self, layer, z):
        return self._layers[layer].activation(z)

    def derivative(self, layer, z):
        return self._layers[layer].derivative(z)

    def getNumLayers(self):
        return self._num_layers

    def getLayerSizes(self):
        return self._layer_sizes

    def getActivationFns(self):
        return self._activations_fns
    def getDerivativeFns(self):
        return self._derivative_fns

