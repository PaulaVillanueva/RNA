import sigmoid
import ej1_data_loader
from layer_model import LayerModel
from feed_forward_solver import NetworkSolver
import functools

loader = ej1_data_loader.Ej1DataLoader()
data = loader.LoadData()
features = data[0] #shape=(333,10)
labels = data[1]

training_data = zip(features, labels)
mini_batch_size = 1
n = len(training_data)
beta = -5
mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]

model =  LayerModel([10,12,1], functools.partial( sigmoid.sigmoid_array,beta),functools.partial( sigmoid.sigmoid_gradient_array,beta))
solver = NetworkSolver(layer_model=model)

lr = 0.005
epochs = 1000
epsilon = 0.05
reg_param = 0.01
solver.learn_minibatch(mini_batches,lr,epochs,epsilon,reg_param)



#mini_batch_size = 10
#b = 2
#model =  LayerModel([10,2,1], functools.partial( sigmoid.sigmoid_array,b),functools.partial( sigmoid.sigmoid_gradient_array,b))
#solver.learn_minibatch(mini_batches,0.001,200,0.1)
