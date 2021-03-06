import sigmoid
import ej2_data_loader
from layer_model import LayerModel
from feed_forward_solver import NetworkSolver
import functools

loader = ej2_data_loader.Ej2DataLoader()
data = loader.LoadData()
features = data[0] #shape=(333,10)
labels = data[1]

all_data = zip(features, labels)
num_training_samples = int(len(all_data) * 0.75)
num_test_samples = len(all_data) - num_training_samples

training_data = all_data[0:num_training_samples - 1]
test_data = all_data[num_training_samples:len(all_data) -1]

mini_batch_size = 1
n = len(training_data)
beta = 0.01

mini_batches_training = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, num_training_samples - 1, mini_batch_size)]

mini_batches_testing = [
                test_data[k:k+mini_batch_size]
                for k in xrange(0, num_test_samples - 1, mini_batch_size)]

model =  LayerModel([8,10,2], functools.partial( sigmoid.sigmoid_array,beta),functools.partial( sigmoid.sigmoid_gradient_array,beta))
solver = NetworkSolver(layer_model=model)

lr = 2.5
epochs = 1000
epsilon = 0.05
reg_param = 0.0
solver.learn_minibatch(mini_batches_training,mini_batches_testing,lr,epochs,epsilon,reg_param)
