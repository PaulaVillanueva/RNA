import numpy as np


# TODO: optimizar y chequear

class HebbianNN:
    def __init__(self, inputs, outputs):

        self._inputs = inputs
        self._outputs = outputs
        self._weights = np.random.uniform(-0.5, 0.5, (inputs, outputs))

    def train(self, ds, epochs, oja=False):
        weights = self._weights
        for e in range(1, epochs + 1):

            lr = 0.1 / e

            for x in ds:
                y = np.dot(x.transpose(), weights)
                dw = np.zeros((self._inputs, self._outputs), dtype=float)

                for j in range(self._outputs):
                    for i in range(self._inputs):
                        xi_mo = 0
                        for k in range(oja and self._outputs or (j + 1)):
                            xi_mo += y.transpose()[k] * weights[i][k]
                        dw[i][j] = lr * y.transpose()[j] * (x[i] - xi_mo)

                weights += dw

        return weights
    
    def orthogonal(self, we, eps):
		dif = np.dot(we.transpose(),we) - np.identity(we.shape[1])
		return (dif < eps).all()
