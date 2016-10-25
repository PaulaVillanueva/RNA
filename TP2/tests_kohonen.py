import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pyl

from kohonen import Kohonen
from heat_map import HeatMap
from data_loader import DataLoader
from mpl_toolkits.mplot3d import Axes3D


loader = DataLoader()
fs, ls =  loader.LoadData("ds/tp2_training_dataset.csv")


epochs = 50
lrcons = 0.07
eps = 0.01


train_fs = fs[:600]


KH = Kohonen((10,10), train_fs.shape[1])
KH.train(train_fs, epochs)
HT = HeatMap()
HT.displayHeatMap((10,10), KH.weights(), train_fs, ls, lambda x : x)

