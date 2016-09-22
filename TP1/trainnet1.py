import argparse
import ej1_data_loader
from model_io import ModelIO
from params_io import ParamsIO
from layer_model import LayerModel
from feed_forward_solver import NetworkSolver
import functools


parser = argparse.ArgumentParser(description='Parametros de la red')


parser.add_argument('-m', metavar='M', type=str,
                    help='Ruta al archivo de modelo (lmodel)', required=True)

parser.add_argument('-o', metavar='O', type=str,
                    help='Ruta del archivo de salida con los pesos', required=True)

parser.add_argument('-t', metavar='T', type=int,
                    help='Epochs', required=True)

parser.add_argument('-e', metavar='E', type=float,
                    help='Epsilon', required=True)

parser.add_argument('-l', metavar='L', type=float,
                    help='Learning rate', required=True)

parser.add_argument('-b', metavar='B', type=int, default=1,
                    help='Size minibatch (default 1)', required=True)

parser.add_argument('-x', metavar='X', type=str, default=1,
                    help='Archivo de samples', required=True)

args = parser.parse_args()


loader = ej1_data_loader.Ej1DataLoader()
data = loader.LoadData(args.x)
features = data[0] #shape=(333,10)
labels = data[1]

all_data = zip(features, labels)
num_training_samples = int(len(all_data) * 0.75)
num_test_samples = len(all_data) - num_training_samples

training_data = all_data[0:num_training_samples - 1]
test_data = all_data[num_training_samples:len(all_data) -1]

mini_batch_size = args.b
n = len(training_data)

mini_batches_training = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, num_training_samples - 1, mini_batch_size)]

mini_batches_testing = [
                test_data[k:k+mini_batch_size]
                for k in xrange(0, num_test_samples - 1, mini_batch_size)]

mloader = ModelIO()
model =  mloader.load_model(args.m)

solver = NetworkSolver(model,weights=model.getInitializedWeightMats(),biases=model.getInitializedBiasVectors())

lr = args.l
epochs =  args.t
epsilon =  args.e
reg_param = 0.0
solver.learn_minibatch(mini_batches_training,mini_batches_testing,lr,epochs,epsilon,reg_param)
pio=ParamsIO()
pio.save_params( args.o , solver._weights, solver._biases)
print "Pesos guardados en ", args.o