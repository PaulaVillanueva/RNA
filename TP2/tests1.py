import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pyl
from us_learning import HebbianNN
from data_loader import DataLoader


loader = DataLoader()
f, l =  loader.LoadData("ds/tp2_training_dataset.csv")

