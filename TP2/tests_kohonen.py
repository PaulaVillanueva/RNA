import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pyl

from params_io import ParamsIO
from preprocessor import FeatureNormalizer
#from kohonen_classifier import KohonenClassifier
from kohonen import Kohonen
from heat_map import HeatMap
from data_loader import DataLoader

from mpl_toolkits.mplot3d import Axes3D


loader = DataLoader()
fs, ls =  loader.LoadData("ds/tp2_training_dataset.csv")


epochs = 1000
lrcons = 0.07
eps = 0.01

train_ls = ls[:600]
#train_fs = FeatureNormalizer().normalize_centered_feautres(fs[:600])
train_fs = fs[:600]

test_fs = fs[600:]
test_ls = ls[600:]

HT = HeatMap()
param_saver = ParamsIO()
KH = Kohonen((15,15), train_fs.shape[1])
plot_hook = lambda : HT.displayHeatMap((15,15), KH.weights(), train_fs, train_ls, lambda x : x)
checkpoint_hook =  lambda : param_saver.save_params("./kohonen.params", KH._output_layout, KH.weights(), KH.get_epoch())
KH.setPlotHook(plot_hook)
KH.setCheckpointHook(checkpoint_hook)
KH.train(train_fs, epochs)
raw_input("Press any key...")

# classifier = KohonenClassifier((15,15), KH.weights(), train_fs, train_ls)
# missclassified = 0
# for x, y in zip(test_fs,test_ls):
#     predicted = classifier.Classify(x)
#     print "X: ", x, " Actual: ", y, " Predicted: ", predicted
#     if (not predicted == y):
#         missclassified = missclassified + 1
#
#     print "Average classification error: ", (len(y) - missclassified)    / len(y)
#
#



