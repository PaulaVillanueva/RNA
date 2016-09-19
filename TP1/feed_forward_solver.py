import numpy as np
from layer_model import LayerModel
import sigmoid


class FeedForwardSolver:

    def __init__(self, weights, layer_model):
        self._weights = weights
        self._solve_batch_and_return_result_array = np.vectorize(self.solve_sample_and_return_activations)
        self._layer_model = layer_model
        self._dw = np.zeros(0)
    # nxm mx1
    # X: 1xm
    # Weight: mxn
    # Z: 1xn
    # Aplicar feed-forward devolviendo la salida de la red para un input dado
    def solve_sample_and_return_activations(self, sample):

        ret = []
        z = sample
        ret.append(z)
        # 1xm x mxn = 1xn
        for w in self._weights:
            # Agrego la unidad de Bias. La ultima unidad calculada  va a salir sin Bias,
            # lo cual esta bien porque es la de salida
            z = [1] + z
            z = self._layer_model.getActivationFn()(np.dot(z, w))
            ret.append(z)

        return ret

    #Aplica solve_sample_and_return_output a un batch completo
    def solve_batch_and_return_result_array(self, batch):
        return self._solve_batch_and_return_result_array(batch)

    #TODO: chequear shapes
    def activation(self, Xh):
        return self.solve_sample_and_return_activations(Xh)
        #y = []
        #L = self._layer_model.get_total_layers()
        #y.append(Xh + [-1]) #con bias
        # 1xm x mxn = 1xn
        #for j in range(1,L):
        #    y.append(self._activation_fn(np.dot(y[j-1], self._weights[j])) )
        #return y

    def correction(self, Zh, Y):
        L = self._layer_model.get_total_layers() - 1
        _dw = np.zeros(np.shape(Zh))
        coef = 0.01 #TODO: hacerlo setteable

        E = Zh - Y[L]
        print("Zh:",Zh)
        print("YL:",Y[L].shape)
        e = np.linalg.norm(E)
        for j in range(L-1, 0, -1):
            print("Error shape: ", np.shape(E))
            print("Yj-1 shape: ", np.shape(Y[j-1]))
            print("wj shape: ", np.shape(self._weights[j]))
            D = E * self._layer_model.getActivationDerivativeFn()(np.dot(Y[j-1], self._weights[j]))
            _dw[:,j] = _dw[:,j] + coef*(np.dot(D, Y[j-1]))
            E = np.dot(D, np.transpose(self._weights[j]))
        return e

    def adaptation(self):
        L =  self._layer_model.get_total_layers()
        for j in range(1, L):
            self._weights[j] = self._weights[j] + self._dw[:,j]
            self._dw[:,j] = 0

    def batch(self,X,Z):
        e = 0
        p = X.shape[0] # p: cant instancias del dataset
        for h in range(0,p-1):
            print("X.shape",X.shape)
            y = self.activation(X[h])
            e = e + self.correction(Z[h], y)
        self.adaptation()
        return e
