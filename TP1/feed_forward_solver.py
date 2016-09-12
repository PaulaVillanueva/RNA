import numpy as np


class FeedForwardSolver:
    def __init__(self, weights, activation_fn):
        self._weights = weights
        #self._activation_fn = np.vectorize(activation_fn)
        self._activation_fn = activation_fn

    # nxm mx1
    # X: 1xm
    # Weight: mxn
    # Z: 1xn
    # Aplicar feed-forward devolviendo la salida de la red para un input dado
    def solve_sample_and_return_output(self, X):
        z = X
        # 1xm x mxn = 1xn
        for w in self._weights:
            # Queda nx1
            z = self._activation_fn(np.dot(z, w))

        return z
