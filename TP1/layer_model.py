import numpy as np
class LayerModel:

    # numInputUnits: Cantidad de unidades de entrada
    # hiddenLayers: Array con la cantidad de hidden layers de la red
    # numOutputLayers: Cantidad de unidades de salida
    def __init__(self, numInputUnits, hiddenLayers, numOutputUnits, activationFn):
        self._numInputUnits = numInputUnits
        self._hiddenLayers = hiddenLayers
        self._numOutputUnits = numOutputUnits
        self._activationFn = activationFn

    #Devuelve una lista de las matrices de pesos inicializadas en random para el modelo
    #Se agregan las unidades de bias en la capa de input y las ocultas
    #tam de lista: #capas-1 (chequear)
    #Cada matriz de pesos: tam ??
    def getInitializedWeightMats(self):
        ret = []
        num_all_layers = [self._numInputUnits + 1] + (self._hiddenLayers + 1) + [self._numOutputUnits] #TypeError: can only concatenate list (not "int") to list
        for i in range(0,len(num_all_layers) - 1):
            weigth_mat = np.random.uniform(-1,1,[num_all_layers[i], num_all_layers[i+1]])
            ret.append(weigth_mat)

        return ret

    def getActivationFn(self):
        return self._activationFn

    def getNumInputUnitsWithBiasUnit(self):
        self._numInputUnits + 1


    def getNumInputUnitsWithoutBiasUnit(self):
        self._numInputUnits


    def getNumHiddenUnitsWithBiasUnit(self):
        self._hiddenLayers + 1


    def getNumHiddenUnitsWithoutBiasUnit(self):
        self._hiddenLayers


    def getNumOutputUnits(self):
        self._numOutputUnits
