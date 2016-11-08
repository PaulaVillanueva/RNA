import numpy as np
import matplotlib.pyplot as plt
from us_learning import HebbianNN
from data_loader import DataLoader
from hbplot import plot3d, plot2d, get_data_x_cat


loader = DataLoader()
fs, ls =  loader.LoadData("ds/tp2_training_dataset.csv")


epochs = 500
lrcons = 0.07
eps = 0.05
noutputs = 9

HB = HebbianNN(len(fs[0]), noutputs, 0.5)
train_fs = fs[:600]

# OjaM

we = HB.train_opt(train_fs, eps, lrcons, epochs, True)

print "check orthogonality: ", HB.orthogonal(we,eps)

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

plot3d(reduced_ds_train, reduced_ds_val,1,2,3)

# plots 2d


data_x_cat = get_data_x_cat(reduced_ds_train)

data_x_cat_val = get_data_x_cat(reduced_ds_val)


for i in range(1,noutputs+1):
    for j in range(i,noutputs+1):
        f, axarr = plt.subplots(1,1)
        plot2d(data_x_cat, data_x_cat_val, axarr, i, j)
        plt.savefig('2dplots/oja/9dim-pc'+str(i)+'-pc'+str(j)+'.png')
        plt.close(f)

rawinput = raw_input("Ingrese 3 enteros separados por espacios correspondientes a componentes principales que desea visualizar: ")
pcs = [int(x) for x in rawinput.split()]

plot3d(reduced_ds_train, reduced_ds_val,pcs[0],pcs[1],pcs[2])

# Sanger

we = HB.train_opt(train_fs, eps, lrcons, epochs)

print "check orthogonality: ", HB.orthogonal(we,eps)

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
plot3d(reduced_ds_train, reduced_ds_val,1,2,3)

## plots 2d

data_x_cat = get_data_x_cat(reduced_ds_train)

data_x_cat_val = get_data_x_cat(reduced_ds_val)


for i in range(1,noutputs+1):
    for j in range(i,noutputs+1):
        f, axarr = plt.subplots(1,1)
        plot2d(data_x_cat, data_x_cat_val, axarr, i, j)
        plt.savefig('2dplots/sanger/9dim-pc'+str(i)+'-pc'+str(j)+'.png')
        plt.close(f)

## plots3d elegidos

rawinput = raw_input("Ingrese 3 enteros separados por espacios correspondientes a componentes principales que desea visualizar: ")
pcs = [int(x) for x in rawinput.split()]

plot3d(reduced_ds_train, reduced_ds_val,pcs[0],pcs[1],pcs[2])