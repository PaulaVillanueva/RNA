import numpy as np
from us_learning import HebbianNN
from data_loader import DataLoader


loader = DataLoader()
fs, ls =  loader.LoadData("ds/tp2_training_dataset.csv")


epochs = 500
lrcons = 0.07
eps = 0.01

HB = HebbianNN(len(fs[0]), 3, 0.5)
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

HB.plot2d(reduced_ds_train, reduced_ds_val)

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

# plots 2d

HB.plot2d(reduced_ds_train, reduced_ds_val)
