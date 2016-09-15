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

	#def activation(self, sample): #TODO: chequear shapes, guardar Y's
		#y = sample + [-1] #con bias
        ## 1xm x mxn = 1xn
        #for j in range(1,LayerModel.getNumHiddenUnitsWithoutBiasUnit()):
            #y = self._activation_fn(np.dot(y, _weights[j])) #_weights pesos de todos??
        #return y
        
    #def correction(self, labels, our_result): #TODO: chequear shapes
		#E = labels - our_result
		#e = np.linalg.norm(E)
		#dw = np.zeros(np.shape(labels))
		#for j in range(LayerModel.getNumHiddenUnitsWithoutBiasUnit(), 0, -1):
			#D = E * sigmoid.sigmoid_gradient_array(np.dot(y[j-1], _weights[j])
			#dw = dw + coef*(np.dot(D, y[j-1])
			#E = np.dot(D, np.transpose(_weights[j]))
		#return e
