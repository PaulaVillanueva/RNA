import argparse
import cPickle
import matplotlib.pyplot as plt
from us_learning import HebbianNN
from data_loader import DataLoader
from hbplot import plot3d, plot2d, get_data_x_cat


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
                    help='Ruta del archivo de entrada con la red', required=True)

parser.add_argument('-x', type=str, default=1,
                    help='Ruta de dataset de validacion', required=True)

args = parser.parse_args()

params_file = args.p
input_file = args.x
HB = load(params_file)

print "Loading network..."

loader = DataLoader()
ds = loader.LoadRawData(input_file)

# Oja
reduced_ds_test = HB.reduce(HB.weights_oja(),ds)
reduced_ds_train = load("oja_reduced_ds")

plot3d(reduced_ds_train, reduced_ds_test,1,2,3)

data_x_cat = get_data_x_cat(reduced_ds_train)
data_x_cat_test = get_data_x_cat(reduced_ds_test)

f, axarr = plt.subplots(1,3)
plot2d(data_x_cat, data_x_cat_test, axarr[0], 1, 2)
plot2d(data_x_cat, data_x_cat_test, axarr[1], 1, 3)
plot2d(data_x_cat, data_x_cat_test, axarr[2], 2, 3)
plt.show()


# Sanger 
reduced_ds_test = HB.reduce(HB.weights_sanger(),ds)
reduced_ds_train = load("sanger_reduced_ds")

plot3d(reduced_ds_train, reduced_ds_test,1,2,3)

data_x_cat = get_data_x_cat(reduced_ds_train)
data_x_cat_test = get_data_x_cat(reduced_ds_test)

f, axarr = plt.subplots(1,3)
plot2d(data_x_cat, data_x_cat_test, axarr[0], 1, 2)
plot2d(data_x_cat, data_x_cat_test, axarr[1], 1, 3)
plot2d(data_x_cat, data_x_cat_test, axarr[2], 2, 3)
plt.show()

