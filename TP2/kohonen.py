import numpy as np
import math

class Kohonen:
    def __init__(self, output_layout, num_input):
        self._output_layout = output_layout
        self._num_input = num_input
        self._weights = None
        # Normalizo el sigma inicial basado en valor experimental calculado para 10x10
        self._sigma0 = 4.5 * ((self._output_layout[0] / 10))
        #self._tau = 50.0

        self._tau = 0.005
        self._epoch = 0


    def train(self, X, epochs):
        self.initialize_weigths()
        for n in range(1,epochs+1):
            self._epoch = n
            self.printH(n)
            for x in X:
                winner = self.get_winner_unit_for_sample (x)
                self.update_neighbordhood_weights (winner, x, n)
            print "Finished epoch ", n
            if n % 50 == 0 and self._plot_hook != None:
                self._plot_hook()
            if n % 50 == 0 and self._checkpoint_hook != None:
                self._checkpoint_hook()

        if self._plot_hook != None:
            self._plot_hook()
        if self._checkpoint_hook != None:
            self._checkpoint_hook()

    def initialize_weigths(self):
        self._weights = {}
        for i in range(self._output_layout[0]):
            for j in range(self._output_layout[1]):
                self._weights[(i,j)] = np.random.rand(self._num_input)

    def get_winner_unit_for_sample (self, sample):
        return min(self._weights.keys(), key=lambda k: np.linalg.norm(self._weights[k] - sample))

    def sigma(self, n):
        #return self._sigma0 * math.exp(-(n / self._tau))
        return self._sigma0 / (1+n*self._sigma0 * self._tau)

    def h(self, n, j, i):
        return math.exp(
            -(
                (self.ddistance(i,j)**2)
                /
                (2*self.sigma(n)**2)
            )
        )
    def update_neighbordhood_weights(self, winner, x, n):

        for u in self._weights.keys():
            hache = self.h(n, u, winner)
            self._weights[u] = self._weights[u] + self.lr(n) * hache * (x - self._weights[u])


    # Distancia Toroide
    def ddistance(self, i, j):
        #TODO:  Me falta calcular esta funcion
        w = self._output_layout[0]
        h = self._output_layout[1]
        min_x = min(i[0], j[0])
        min_y = min(i[1], j[1])
        max_x = max(i[0], j[0])
        max_y = max(i[1], j[1])

        dist_x = min(max_x-min_x, min_x + w - max_x)
        dist_y = min(max_y - min_y, min_y + h - max_y)

        return math.sqrt(dist_x**2 + dist_y**2)

    def lr(self, n):
        return 0.01

    def weights(self):
        return self._weights

    def printH(self,n):
        nb = 0
        for i in range(self._output_layout[0]):
            for j in range(self._output_layout[1]):
                    hache = self.h(n, (5,5), (i,j))
                    print "H:", hache
                    if hache > 0.05 :
                        nb = nb + 1
        print "Vecinos: ", nb


    def plotKook(self, hook):
        return self._plot_hook

    def setPlotHook(self, hook):
        self._plot_hook = hook

    def setCheckpointHook(self, hook):
        self._checkpoint_hook = hook




    def get_epoch(self):
        return self._epoch
