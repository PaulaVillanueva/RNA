import argparse
import cPickle
import numpy as np
from us_learning import HebbianNN
from data_loader import DataLoader


def saveAs(saveIn, obj):
	print "Guardando en %s" % saveIn
	with open(saveIn, 'wb') as f:
		cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)

def load(loadFrom):
	print "Cargando desde %s" % loadFrom
	with open(loadFrom, 'rb') as f:
		return cPickle.load(f)


print "Hebbian Learning"
parser = argparse.ArgumentParser(description='Parametros de la red')


parser.add_argument('-p', type=str,
                    help='Ruta del archivo de salida para guardar la red', required=True)

parser.add_argument('-x', type=str, 
                    help='Ruta de dataset de entrenamiento', required=True)

parser.add_argument('-n', type=int, default=100,
                    help='Cantidad maxima de epochs', required=False)

args = parser.parse_args()

params_file = args.p
input_file = args.x
epochs = args.n

loader = DataLoader()
fs, ls =  loader.LoadData(args.x)

train_ls = ls
train_fs = fs


lrcons = 0.07
eps = 0.05

HB = HebbianNN(len(fs[0]), 3, 0.5)

# Oja

we_oja = HB.train_opt(train_fs, eps, lrcons, epochs, True)

outputs = np.array([np.dot(x.transpose(), we_oja) for x in train_fs])
reduced_ds_train = [ [data[0]] + data[1].tolist() for data in zip(train_ls,outputs) ]
saveAs("oja_reduced_ds", reduced_ds_train)

# Sanger

we_sanger = HB.train_opt(train_fs, eps, lrcons, epochs, False)

outputs = np.array([np.dot(x.transpose(), we_sanger) for x in train_fs])
reduced_ds_train = [ [data[0]] + data[1].tolist() for data in zip(train_ls,outputs) ]
saveAs("sanger_reduced_ds", reduced_ds_train)

###### 

saveAs(params_file, HB)

raw_input("Press any key...")


