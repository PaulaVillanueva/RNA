import numpy as np


class FeedForwardSolver:
    def __init__(self, weights, activation_fn):
        self._weights = weights
        #self._activation_fn = np.vectorize(activation_fn)
        self._activation_fn = activation_fn
        self._solve_batch_and_return_result_array = np.vectorize(self.solve_sample_and_return_output)
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

