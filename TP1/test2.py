import sigmoid
import ej1_data_loader
from TP1.model_io import ModelIO
from TP1.params_io import ParamsIO
from layer_model import LayerModel
from feed_forward_solver import NetworkSolver
import functools

loader = ej1_data_loader.Ej1DataLoader()
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
beta = 5

mini_batches_training = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, num_training_samples - 1, mini_batch_size)]

mini_batches_testing = [
                test_data[k:k+mini_batch_size]
                for k in xrange(0, num_test_samples - 1, mini_batch_size)]

#model =  LayerModel([10,12,1], functools.partial( sigmoid.sigmoid_array,beta),functools.partial( sigmoid.sigmoid_gradient_array,beta))
mloader = ModelIO()
model =  mloader.load_model("/home/berna/PycharmProjects/RNA/TP1/models/ej1.lmodel")

#solver = NetworkSolver(layer_model=model,weights=model.getInitializedWeightMats(),biases=model.getInitializedBiasVectors())
solver = NetworkSolver(model,weights=model.getInitializedWeightMats(),biases=model.getInitializedBiasVectors())

lr = 0.005
epochs = 100
epsilon = 0.05
reg_param = 0.0
solver.learn_minibatch(mini_batches_training,mini_batches_testing,lr,epochs,epsilon,reg_param)
pio=ParamsIO()
pio.save_params("./prueba.params", solver._weights, solver._biases)
