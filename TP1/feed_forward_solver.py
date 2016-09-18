import numpy as np
from layer_model import LayerModel
import sigmoid


class FeedForwardSolver:

    def __init__(self, weights, activation_fn, layer_model):
        self._weights = weights
        #self._activation_fn = np.vectorize(activation_fn)
        self._activation_fn = activation_fn
        self._solve_batch_and_return_result_array = np.vectorize(self.solve_sample_and_return_output)
        self._layer_model = layer_model
        self._dw = np.zeros(0)
    # nxm mx1
    # X: 1xm
    # Weight: mxn
    # Z: 1xn
    # Aplicar feed-forward devolviendo la salida de la red para un input dado
    def solve_sample_and_return_output(self, sample):

        z = sample
        # 1xm x mxn = 1xn
        for w in self._weights:
            # Agrego la unidad de Bias. La ultima unidad calculada  va a salir sin Bias,
            # lo cual esta bien porque es la de salida
            z = [1] + z
            z = self._activation_fn(np.dot(z, w))

        return z

    #Aplica solve_sample_and_return_output a un batch completo
    def solve_batch_and_return_result_array(self, batch):
        return self._solve_batch_and_return_result_array(batch)

    #TODO: chequear shapes
    def activation(self, Xh):
        y = []
        L = LayerModel.getNumHiddenLayers()+2 
        y.append(Xh + [-1]) #con bias
        # 1xm x mxn = 1xn
        for j in range(1,L):
            y.append(self._activation_fn(np.dot(y[j-1], _weights[j])) )
        return y[L]

    def correction(self, Zh, y):
        L = LayerModel.getNumHiddenLayers()+2
        _dw = np.zeros(np.shape(Zh))
        coef = 0.01 #TODO: hacerlo setteable

        E = Zh - y[L]
        e = np.linalg.norm(E)
        for j in range(L, 0, -1):
            D = E * sigmoid.sigmoid_gradient_array(np.dot(y[j-1], _weights[j]))
            _dw[:,j] = _dw[:,j] + coef*(np.dot(D, y[j-1]))
            E = np.dot(D, np.transpose(_weights[j]))
        return e

    def adaptation(self):
        L = LayerModel.getNumHiddenLayers()+2 
        for j in range(1, L):
            _weights[j] = _weights[j] + _dw[:,j]
            _dw[:,j] = 0

    def batch(self,X,Z):
        p = X.shape[0] # p: cant instancias del dataset
        for h in range(1,p):
            self.activation(X[h])
            e = e + self.correction(Z[h])
        self.adaptation()
        return e