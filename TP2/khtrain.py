import argparse
from kohonen_category_mapper import KohonenCategoryMapper
from params_io import ParamsIO
from kohonen import Kohonen
from heat_map import HeatMap
from data_loader import DataLoader

from mpl_toolkits.mplot3d import Axes3D


print "Kohonen Training by P. Villanueva, V. Uhrich, B. Panarello"
parser = argparse.ArgumentParser(description='Parametros de la red')


parser.add_argument('-p', type=str,
                    help='Ruta del archivo de salida para escribir pesos', required=True)

parser.add_argument('-x', type=str, default=1,
                    help='Archivo de training features', required=True)

parser.add_argument('-w', type=int, default=1,
                    help='Ancho del arreglo de neuronas de salida', required=True)

parser.add_argument('-l', type=int, default=1,
                    help='Alto del arreglo de neuronas de salida', required=True)

parser.add_argument('-n', type=int, default=1,
                    help='Cantidad de epochs a entrenar', required=True)


args = parser.parse_args()

params_file = args.p
input_file = args.x


loader = DataLoader()
fs, ls =  loader.LoadData(args.x)


epochs = args.n

train_ls = ls
train_fs = fs

layout = (args.w,args.l)
HT = HeatMap()
param_saver = ParamsIO()
KH = Kohonen(layout, train_fs.shape[1])
plot_hook = lambda : HT.displayHeatMap(layout, KH.weights(), train_fs, train_ls)
checkpoint_hook =  lambda : param_saver.save_params(args.p, KH._output_layout, KH.weights(), KohonenCategoryMapper().getCategoryMap(layout, KH.weights(), train_fs, train_ls ) ,  KH.get_epoch())
KH.setPlotHook(plot_hook)
KH.setCheckpointHook(checkpoint_hook)
KH.train(train_fs, epochs)


HT.displayHeatMap(layout, KH.weights(), train_fs, train_ls, False)

raw_input("Press any key...")


