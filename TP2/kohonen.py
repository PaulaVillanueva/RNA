import numpy as np
import math

class Kohonen:
    def __init__(self, output_layout, num_input):
        self._output_layout = output_layout
        self._num_input = num_input
        self._weights = None
        self._sigma0 = 1
        self._tau = 1

    def train(self, X):
        self.initialize_weigths()
        for n in range(90):
            for x in X:
                winner = self.get_winner_unit_for_sample (x)
                self.update_neighbordhood_weights (winner, x, n)

    def initialize_weigths(self):
        self._weights = {}
        for unit in self._output_layout:
            self._weights[unit] = np.random.rand(self._num_input)

    def get_winner_unit_for_sample (self, sample):
        return min(self._weights.keys, key=lambda k: np.linalg.norm(self._weights[k] - sample))

    def sigma(self, n):
        return self._sigma0 * math.exp(-n / self._tau)

    def h(self, n, j, i):
        return math.exp(-(self.ddistance(i,j)**2)/2*self.sigma(n)**2)

    def update_neighbordhood_weights(self, winner, x, n):
        for u in self._weights.keys:
            self._weights[u] = self._weights[u] + self.lr(n) * self.h(n, u, winner) * (x - self._weights[u])

    def ddistance(self, i, j):
        #TODO:  Me falta calcular esta funcion
        pass
