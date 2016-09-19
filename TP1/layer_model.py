import numpy as np
class LayerModel:

    # numInputUnits: Cantidad de unidades de entrada
    # hiddenLayers: Array con la cantidad de hidden layers de la red
    # numOutputLayers: Cantidad de unidades de salida
    def __init__(self, numInputUnits, hiddenLayers, numOutputUnits, activationFn, activationDerivativeFunction):
        self._numInputUnits = numInputUnits
        self._unitsPerHiddenLayer = np.array([hiddenLayers])
        self._numOutputUnits = numOutputUnits
        self._activationFn = activationFn
        self._activationDerivativeFunction = activationDerivativeFunction
        self._numUnitsPerLayer =np.array([numInputUnits] + hiddenLayers + [numOutputUnits])
        self._numHiddenLayers = len(hiddenLayers)

    #Devuelve una lista de las matrices de pesos inicializadas en random para el modelo
    #Se agregan las unidades de bias en la capa de input y las ocultas
    #tam de lista: #capas-1 (chequear)
    #Cada matriz de pesos: tam ??
    def getInitializedWeightMats(self):
        ret = []

        num_all_layers = self.get_total_layers()
        for i in range(0,num_all_layers - 1):
            weigth_mat = np.random.uniform(-1,1,[self._numUnitsPerLayer[i], self._numUnitsPerLayer[i+1]])
            ret.append(weigth_mat)

        return ret
        
    def getInitializedDeltaWMats(self):
        ret = []

        num_all_layers = self.get_total_layers()
        for i in range(0,num_all_layers - 1):
            weigth_mat = np.random.uniform(-1,1,[self._numUnitsPerLayer[i], self._numUnitsPerLayer[i+1]])
            ret.append(weigth_mat)

        return ret

    def get_total_layers(self):
        return self._numHiddenLayers + 2


    def getActivationFn(self):
        return self._activationFn

    def getActivationDerivativeFn(self):
        return self._activationDerivativeFunction

    def getNumInputUnitsWithBiasUnit(self):
        self._numInputUnits + 1


    def getNumInputUnitsWithoutBiasUnit(self):
        self._numInputUnits


    def getNumHiddenUnitsWithBiasUnit(self):
        self._unitsPerHiddenLayer + 1


    def getNumHiddenUnitsWithoutBiasUnit(self):
        self._unitsPerHiddenLayer


    def getNumOutputUnits(self):
        self._numOutputUnits

    def getNumHiddenLayers(self):
        return self._unitsPerHiddenLayer
