import numpy as np
from feed_forward_solver import NetworkSolver


class GradientSolver:
    def __init__(self, batch_size, activation_fn, activation_gradient_fn,model):
        self._batch_size = batch_size
        self._activation_fn = activation_fn
        self._activation_gradient_fn = activation_gradient_fn
        self._model = model
        self._fforwardFn

    def get_gradient(self, features, labels, weights):
        ff_solver = NetworkSolver(weights, self._activation_fn)
        features_batch = features(0, self._batch_size - 1)
        labels_batch = labels(0, self._batch_size - 1)
        Y = ff_solver.solve_batch_and_return_result_array(features_batch)
        output_delta = labels_batch - Y

        for j in range(0, self._batch_size - 1):
            batch_delta = self.init_batch_delta(output_delta[j])
            #TODO: CALCULAR LOS DELTA PARA CADA UNIDAD
            #TODO: ACUMULAR LOS DELTA Y PROMEDIAR POR NUMERO DE SAMPLES. CON ESTO CALCULAR LAS DERIVADAS PARCIALES

    def init_batch_delta(self, output_delta):
        ret = []
        ret.append(np.array(self._model.getNumInputUnitsWithoutBiasUnit()))
        for n in self._model.getNumHiddenUnitsWithoutBiasUnit():
            ret.append(np.array(n))
        ret.append(output_delta)
        return ret