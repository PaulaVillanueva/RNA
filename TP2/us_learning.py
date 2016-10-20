import numpy as np


# TODO: optimizar

class HebbianNN:
    def __init__(self, inputs, outputs, wint):

        self._inputs = inputs
        self._outputs = outputs
        self._weights = np.random.uniform(-wint, wint, (inputs, outputs))

    def train_wepochs(self, ds, epochs, oja=False):
        weights = self._weights
        for e in range(1, epochs + 1):

            lr = 0.0001 / e

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
        
    def train(self, ds, eps, lrcons, max_epochs, oja=False):
        weights = np.matrix.copy(self._weights)
        e = 0
        while (not self.orthogonal(weights,eps)) and e < max_epochs:
            e+=1
            lr = lrcons * (e**-2)

            for x in ds:
                y = np.dot(x, weights)
                dw = np.zeros((self._inputs, self._outputs), dtype=float)

                for j in range(self._outputs):
                    for i in range(self._inputs):
                        xi_mo = 0
                        for k in range(oja and self._outputs or (j + 1)):
                            xi_mo += y.transpose()[k] * weights[i][k]
                        dw[i][j] = lr * y.transpose()[j] * (x[i] - xi_mo)

                weights += dw
        print "epocas: ", e
        return weights
    

    def train_opt(self, ds, eps, lrcons, max_epochs, oja=False):
        weights = np.matrix.copy(self._weights)
        e = 0
        if not oja:
            dim = (self._outputs, self._outputs)
            U = np.triu(np.ones(dim))
        while (not self.orthogonal(weights,eps)) and e < max_epochs:
            e+=1
            lr = lrcons * (e**-2)

            for x in ds:
                x = np.array([x])
                y = np.dot(x, weights)
                x_mo = np.zeros(x.shape)
                dw = np.zeros((self._inputs, self._outputs), dtype=float)

                if oja:
                    x_mo[:] = np.dot(y, weights.transpose())
                    dw[:,:] = lr * np.dot((x - x_mo).transpose(), y)
                else:
                    xy = np.dot(x.transpose(), y)
                    x_moy = np.dot(np.tril(np.dot(y.transpose(), y)), weights.transpose()).transpose()
                    dw[:,:] = lr * ( xy - x_moy )

                weights += dw
        print "epocas: ", e
        return weights


    def orthogonal(self, we, eps):
        dif = np.dot(we.transpose(),we) - np.identity(we.shape[1])
        return (np.abs(dif) < eps).all()

    def norm_eq1(self, we, eps):
        norms = np.linalg.norm(we, axis=0)
        return (np.abs(norms - np.ones(norms.shape)) < eps).all()

    def activate(self, we, ds):
        return [ [data[0]] + np.dot(data[1:], we).tolist() for data in ds ]