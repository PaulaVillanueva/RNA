import numpy as np
import matplotlib.pyplot as plt
from us_learning import HebbianNN
from data_loader import DataLoader


loader = DataLoader()
fs, ls =  loader.LoadData("ds/tp2_training_dataset.csv")


epochs = 500
lrcons = 0.07
eps = 0.05

HB = HebbianNN(len(fs[0]), 9, 0.5)
train_fs = fs[:600]

# OjaM

we = HB.train_opt(train_fs, eps, lrcons, epochs, True)

print "check orthogonality: ", HB.orthogonal(we,eps)
print "check norm == 1: ", HB.norm_eq1(we,eps)

outputs = np.array([np.dot(x.transpose(), we) for x in train_fs])
print "outputs: "
print "mean: ", np.mean(outputs, axis=0)
print "std: ", np.std(outputs, axis=0)
print "var: ", np.var(outputs, axis=0)


# plot 

reduced_ds_train = [ [data[0]] + data[1].tolist() for data in zip(ls[:600],outputs) ]

val_fs = fs[600:]
outputs_val = np.array([np.dot(x.transpose(), we) for x in val_fs])
reduced_ds_val = [ [data[0]] + data[1].tolist() for data in zip(ls[600:],outputs_val) ]

# 3d

HB.plot3d(reduced_ds_train, reduced_ds_val)

# plots 2d

data_x_cat = HB.get_data_x_cat(reduced_ds_train)

data_x_cat_val = HB.get_data_x_cat(reduced_ds_val)

f, axarr = plt.subplots(9, 9)
for i in range(1,10):
    for j in range(1,10):
        HB.plot2d(data_x_cat, data_x_cat_val, axarr[i-1, j-1], i, j)

plt.legend(numpoints=1,ncol=6)

plt.show()

# Sanger

we = HB.train_opt(train_fs, eps, lrcons, epochs)

print "check orthogonality: ", HB.orthogonal(we,eps)
print "check norm == 1: ", HB.norm_eq1(we,eps)

outputs = np.array([np.dot(x.transpose(), we) for x in train_fs])
print "outputs: "
print("mean: ", np.mean(outputs, axis=0))
print("std: ", np.std(outputs, axis=0))
print("var: ", np.var(outputs, axis=0))


# plot 

reduced_ds_train = [ [data[0]] + data[1].tolist() for data in zip(ls[:600],outputs) ]

val_fs = fs[600:]
outputs_val = np.array([np.dot(x.transpose(), we) for x in val_fs])
reduced_ds_val = [ [data[0]] + data[1].tolist() for data in zip(ls[600:],outputs_val) ]

# 3d
HB.plot3d(reduced_ds_train, reduced_ds_val)

## plots 2d

data_x_cat = HB.get_data_x_cat(reduced_ds_train)

data_x_cat_val = HB.get_data_x_cat(reduced_ds_val)

f, axarr = plt.subplots(9, 9)
for i in range(1,10):
    for j in range(1,10):
        HB.plot2d(data_x_cat, data_x_cat_val, axarr[i-1, j-1], i, j)

plt.legend(numpoints=1,ncol=6)

plt.show()
