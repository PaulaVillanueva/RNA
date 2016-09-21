import numpy as np
from layer_model import LayerModel
import sigmoid


class NetworkSolver:

    def __init__(self, layer_model):
        self._weights = layer_model.getInitializedWeightMats()
        self._biases = layer_model.getInitializedBiasVectors()
        self._layer_model = layer_model


    def do_activation(self, sample):

        aa = [np.reshape(sample, (len(sample), 1))]
        zz = []
        # Bias
        for b, w in zip(self._biases, self._weights):
            z = np.dot(w, aa[-1]) + b
            a = self._layer_model.getActivationFn()(z)
            zz.append(z)
            aa.append(a)
        return (aa,zz)


    def do_backprop_and_return_grad(self, x, y):
        grad_w = self._layer_model.getZeroDeltaW()
        grad_b = self._layer_model.getZeroDeltaB()

        # feedforward
        activations, zs = self.do_activation(x)
        delta = (activations[-1] - y) * self._layer_model.getActivationDerivativeFn()(zs[-1])
        grad_b[-1] = delta
        grad_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in xrange(2, self._layer_model.getNumLayers()):
            z = zs[-l]
            sp = self._layer_model.getActivationDerivativeFn()(z)
            delta = np.dot(self._weights[-l+1].transpose(), delta) * sp
            grad_b[-l] = delta
            grad_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (grad_b, grad_w)



    def correction_mini_batch(self, mini_batch, lr, n, lmbda=0.0):
        """
        lr = learning rate
        lmbda = regularization parameter
        n = longitud del training set
        """
        grad_b = [np.zeros(b.shape) for b in self._biases]
        grad_w = [np.zeros(w.shape) for w in self._weights]
        for x, y in mini_batch:
            delta_grad_b, delta_grad_w = self.do_backprop_and_return_grad(x, y)
            grad_b = [gb+deltagb for gb, deltagb in zip(grad_b, delta_grad_b)]
            grad_w = [gw+deltagw for gw, deltagw in zip(grad_w, delta_grad_w)]
        #lmbda=regularization parameter
        #si lmbda=0.0 hace lo mismo que el clasico
        self._weights = [(1.0 - lr*(lmbda/n)) * w - (lr/len(mini_batch)) * gw
                        for w, gw in zip(self._weights, grad_w)]
        #clasico sin regularizacion
        #    self._weights = [w - (lr / len(mini_batch)) * gw
        #                for w, gw in zip(self._weights, grad_w)]
        self._biases = [b - (lr / len(mini_batch)) * gb
                       for b, gb in zip(self._biases, grad_b)]


    def learn_minibatch(self, mini_batches, lr, epochs, epsilon, lmbda=0.0):
        """si lmbda no se especifica no se usa regularizacion"""
        T = epochs
        t = 0
        e = 999
        n = sum([len(mbatch) for mbatch in mini_batches])
        while e > epsilon and t < T:
            for b in mini_batches:
                self.correction_mini_batch(b, lr, n, lmbda)
            t = t + 1
            e = self.get_training_error(mini_batches, False)
            print ("Error: ", e)

        e = self.get_training_error(mini_batches, True)

    def get_training_error(self, mini_batches, bprint):
        e = 0
        cant = 0
        for b in mini_batches:

            for x, y in b:
                cant = cant + 1
                aa, zz = self.do_activation(x)

                e = e + np.linalg.norm(aa[-1] - y)
                if bprint:
                    print e / cant, np.linalg.norm(aa[-1] - y),aa[-1][0][0], y[0]

        return e / cant

    def get_hits(self, test_data):
        """Devuelve el numero de aciertos de inputs de test para los que
        los outputs que devuelve la red son correctos.
        Con una neurona, el resultado es el mas cercano entre 0 y 1
        al resultado que devolvio la red"""
        test_results = [(self.get_result(self.do_activation(x)[0][-1]), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def get_result(self, act):
        return np.argmin([abs(act-0), abs(act-1)])