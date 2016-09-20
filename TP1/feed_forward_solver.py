import numpy as np
from layer_model import LayerModel
import sigmoid


class FeedForwardSolver:

    def __init__(self, weights, layer_model):
        self._weights = weights
        self._solve_batch_and_return_result_array = np.vectorize(self.solve_sample_and_return_activations)
        self._layer_model = layer_model
        self._dw = self._layer_model.getInitializedDeltaWMats()
    # nxm mx1
    # X: 1xm
    # Weight: mxn
    # Z: 1xn
    # Aplicar feed-forward devolviendo la salida de la red para un input dado
    def solve_sample_and_return_activations(self, sample):

        ret = []
        # Bias
        z = np.insert(sample,0,-1,axis=1)
        ret.append(z)
        i = 0
        # 1xm x mxn = 1xn
        for w in self._weights:
            i = i + 1
            # Agrego la unidad de Bias. La ultima unidad calculada  va a salir sin Bias,
            # lo cual esta bien porque es la de salida
            z = self._layer_model.getActivationFn()(np.dot(z, w))
            if i < len(self._weights):
                z = np.insert(z,0,-1,axis=1)
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
        L = self._layer_model.get_total_layers()

        coef = 0.1 #TODO: hacerlo setteable

        E = Zh - Y[L-1]
        e = np.linalg.norm(E)
        for j in range(L, 1, -1):
            dotprod = np.dot(Y[j-2], self._weights[j-2])
            D = E * self._layer_model.getActivationDerivativeFn()(dotprod)
            self._dw[j-2] = self._dw[j-2] + coef*(np.dot(np.transpose(Y[j-2]), D))
            E = np.dot(D, np.transpose(np.delete(self._weights[j-2],1,0)))
        return e

    def adaptation(self):
        L =  self._layer_model.get_total_layers() - 1
        for j in range(1, L):
            self._weights[j] = self._weights[j] + self._dw[j]
            self._dw[j] = 0

    def batch(self,X,Z):
        e = 0
        p = X.shape[0] # p: cant instancias del dataset
        f = X.shape[1]
        for h in range(0,p-1):
            #print("Xh.shape",X[h].reshape((f,1)).shape)
            y = self.activation(X[h].reshape((1,f)))
            e = e + self.correction(Z[h], y)
        self.adaptation()
        return e
