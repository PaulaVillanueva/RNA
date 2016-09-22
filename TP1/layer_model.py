import numpy as np
class LayerModel:

    # numInputUnits: Cantidad de unidades de entrada
    # hiddenLayers: Array con la cantidad de hidden layers de la red
    # numOutputLayers: Cantidad de unidades de salida
    def __init__(self, layerSizes, activationFn, activationDerivativeFunction):
        self._activationFn = activationFn
        self._activationDerivativeFunction = activationDerivativeFunction

        self._biases = [np.random.randn(y, 1) / 1000 for y in layerSizes[1:]]
        self._weights = [np.random.randn(y, x) / 1000
                        for x, y in zip(layerSizes[:-1], layerSizes[1:])]

        self._layer_sizes = layerSizes
        self._num_layers = len(layerSizes)

    def getInitializedWeightMats(self):
        return self._weights

    def getInitializedBiasVectors(self):
        return self._biases

    def getZeroDeltaW(self):
        return [np.zeros(w.shape) for w in self._weights]

    def getZeroDeltaB(self):
        return [np.zeros(b.shape) for b in self._biases]

    def getActivationFn(self):
        return self._activationFn

    def getActivationDerivativeFn(self):
        return self._activationDerivativeFunction

    def getNumLayers(self):
        return self._num_layers

    def getLayerSizes(self):
        return self._layer_sizes



