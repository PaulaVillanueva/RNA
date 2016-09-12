import numpy as np
class LayerModel:

    # numInputUnits: Cantidad de unidades de entrada
    # hiddenLayers: Array con la cantidad de hidden layers de la red
    # numOutputLayers: Cantidad de unidades de salida
    def __init__(self, numInputUnits, hiddenLayers, numOutputUnits):
        self._numInputUnits = numInputUnits
        self._hiddenLayers = hiddenLayers
        self._numOutputUnits = numOutputUnits

    #Devuelve una lista de las matrices de pesos inicializadas en random para el modelo
    def getInitializedWeightMats(self):
        ret = []
        num_all_layers = [self._numInputUnits] + self._hiddenLayers + [self._numOutputUnits]
        for i in range(0,len(num_all_layers) - 1):
            weigth_mat = np.random.rand(num_all_layers[i], num_all_layers[i+1])
            ret.append(weigth_mat)

        return ret


