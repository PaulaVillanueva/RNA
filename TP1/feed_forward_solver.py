import numpy as np
from layer_model import LayerModel
import sigmoid


class NetworkSolver:

    def __init__(self, layer_model, weights, biases, checkpoint_fn=None):
        self._weights = weights
        self._biases = biases
        self._layer_model = layer_model
        self._check_point_fn=checkpoint_fn

    def do_activation(self, sample):
        aa = [np.reshape(sample, (len(sample), 1))]
        zz = []
        # Bias
        l = 0
        for b, w in zip(self._biases, self._weights):
            l = l + 1
            z = np.dot(w, aa[-1]) + np.reshape(b, (len(b),1)) #no anda para b=1  # es que no deberia ser b=1 tendria que ser b=[1]
            #z = np.dot(w, aa[-1]) + b
            a = self._layer_model.getActivationFns()[l](z)
            zz.append(z)
            aa.append(a)
        return (aa,zz)


    def do_backprop_and_return_grad(self, x, y):
        grad_w = self._layer_model.getZeroDeltaW()
        grad_b = self._layer_model.getZeroDeltaB()

        # feedforward
        activations, zs = self.do_activation(x)
        delta = (activations[-1] - np.reshape(y, (len(y), 1))) * self._layer_model.getDerivativeFns()[-1](zs[-1])
        grad_b[-1] = delta
        grad_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in xrange(2, self._layer_model.getNumLayers()):
            z = zs[-l]
            sp = self._layer_model.getDerivativeFns()[-l](z)
            delta = np.dot(self._weights[-l+1].transpose(), delta) * sp
            grad_b[-l] = delta
            grad_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (grad_b, grad_w)



    def correction_mini_batch(self, mini_batch, lr, n, lmbda=0.0):
        """
        :param mini_batch: set de entrenamiento
        :param lr: learning rate
        :param n: cantidad de samples
        :param lmbda: regularization parameter
        """
        grad_b = [np.zeros(b.shape) for b in self._biases]
        grad_w = [np.zeros(w.shape) for w in self._weights]
        for x, y in mini_batch:
            delta_grad_b, delta_grad_w = self.do_backprop_and_return_grad(x, y)
            grad_b = [gb+deltagb for gb, deltagb in zip(grad_b, delta_grad_b)]
            grad_w = [gw+deltagw for gw, deltagw in zip(grad_w, delta_grad_w)]

        self._weights = [(1.0 - lr*(lmbda/n)) * w - (lr/len(mini_batch)) * gw
                        for w, gw in zip(self._weights, grad_w)]
        #clasico sin regularizacion
        #    self._weights = [w - (lr / len(mini_batch)) * gw
        #                for w, gw in zip(self._weights, grad_w)]
        self._biases = [b - (lr / len(mini_batch)) * gb
                       for b, gb in zip(self._biases, grad_b)]


    def learn_minibatch(self, mini_batches, mini_batches_testing, lr, epochs, epsilon, lmbda=0.0):
        """
        :param mini_batches: set de entrenamiento
        :param mini_batches_testing: set de testing
        :param lr: learning rate
        :param epochs: cantidad de epocas
        :param epsilon: cota de error
        :param lmbda: parametro de regularizacion, si no se especifica, no se regulariza
        imprime errores de entrenamiento y testing por cada epoca
        """
        T = epochs
        t = 0
        e = 999
        n = sum([len(mbatch) for mbatch in mini_batches])
        while e > epsilon and t < T:
            for b in mini_batches:
                self.correction_mini_batch(b, lr, n, lmbda)
            t = t + 1
            e = self.get_prediction_error(mini_batches, False)
            et = self.get_prediction_error(mini_batches_testing, False)
            print "[",t,"] ", "Training Error: ", e, "Val error:", et
            if t % 1000 == 0 and self._check_point_fn is not None:
                print "Saving checkpoint..."
                self._check_point_fn()

        #e = self.get_prediction_error(mini_batches, True)


    def get_prediction_error(self, mini_batches, bprint):
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

    def predict(self, batch, threshold):
        e = 0
        hits = 0.0
        for x, y in batch:
            aa, zz = self.do_activation(x)
            e = e + np.linalg.norm(aa[-1][0] - y)
            if aa[-1][0] < threshold:
                hits += int(y[0] == 0.0)
            else:
                hits += int(y[0] == 1.0)
            print "Original: ", y[0], "Predicted:",aa[-1][0], "Error:", np.linalg.norm(aa[-1][0] - y)

        print "Hit rate: ", hits / len(batch) * 100, "%"
        return e / len(batch)

    def predict_linear(self, batch):
        e = 0
        re = 0
        for x, y in batch:
            aa, zz = self.do_activation(x)
            abs_error = np.linalg.norm(np.transpose(aa[-1]) - y)
            e = e + abs_error
            rel_error = abs_error / np.linalg.norm(y)
            re = re + rel_error

            print "Original: ", y, "Predicted:",np.transpose(aa[-1]), "AbsError:", abs_error, "Relative error: ", rel_error * 100 , "%"


        return "Average absolute Error: ", e / len(batch), " -- Average relative Error: ", re / len(batch)


    def get_hits(self, test_data):
        """
        :param test_data: set de datos de testing
        :return: Devuelve el numero de aciertos de inputs de test para los que
        los outputs que devuelve la red son correctos.
        """
        test_results = [(self.get_result(self.do_activation(x)[0][-1]), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def get_result(self, act):
        """
        :param act: resultado de nuestra red
        :return: el resultado es el mas cercano al resultado
        que devolvio la red entre 0 y 1
        """
        return np.argmin([abs(act-0), abs(act-1)])