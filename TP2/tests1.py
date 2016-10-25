import numpy as np
import matplotlib.pyplot as plt
from us_learning import HebbianNN
from data_loader import DataLoader
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict


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

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', '0.4']
labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9']

# 3d

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for data in reduced_ds_train:
    ax.scatter([data[1]], [data[2]], [data[3]], marker='o', c=colors[int(data[0]) - 1])
    pass

for data in reduced_ds_val:
    ax.scatter([data[1]], [data[2]], [data[3]], marker='d', c=colors[int(data[0]) - 1])
    pass


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

colors = ['b', 'g', 'r', 'c', 'm', 'gold', 'k', 'pink', '0.8']

# 3d

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

data_x_cat = defaultdict(dict)
for data in reduced_ds_train:
    cat = int(data[0])
    if not data_x_cat[cat]:
        data_x_cat[cat] = defaultdict(list)

    data_x_cat[cat][1].append(data[1])
    data_x_cat[cat][2].append(data[2])
    data_x_cat[cat][3].append(data[3])

cats = ['Categoria1','Categoria2','Categoria3','Categoria4','Categoria5','Categoria6','Categoria7','Categoria8','Categoria9']
for c in range(0,9):
    ax.plot([], [], marker='o', color=colors[c], label='categoria '+ str(c + 1) + ' training')
    ax.scatter(data_x_cat[c+1][1], data_x_cat[c+1][2], data_x_cat[c+1][3], marker='o', color=colors[c], label='categoria '+ str(c + 1))

data_x_cat = defaultdict(dict)
for data in reduced_ds_val:
    cat = int(data[0])
    if not data_x_cat[cat]:
        data_x_cat[cat] = defaultdict(list)

    data_x_cat[cat][1].append(data[1])
    data_x_cat[cat][2].append(data[2])
    data_x_cat[cat][3].append(data[3])


for c in range(0,9):
    ax.plot([], [], marker='x', color=colors[c], label='categoria '+ str(c + 1) + ' validation')
    ax.scatter(data_x_cat[c+1][1], data_x_cat[c+1][2], data_x_cat[c+1][3], marker='x', color=colors[c], label='categoria '+ str(c + 1))

plt.legend(numpoints=1,ncol=6)

plt.show()

# plots 2d

fig = plt.figure()
ax = fig.add_subplot(131)

for data in reduced_ds_train:
    ax.scatter([data[1]], [data[2]], marker='x', c=colors[int(data[0]) - 1])
    pass

ax = fig.add_subplot(132)

for data in reduced_ds_train:
    ax.scatter([data[2]], [data[3]], marker='x', c=colors[int(data[0]) - 1])
    pass

ax = fig.add_subplot(133)

for data in reduced_ds_train:
    ax.scatter([data[3]], [data[1]], marker='x', c=colors[int(data[0]) - 1])
    pass

# for data in reduced_ds_val:
#     ax.scatter([data[1]], [data[2]], marker='d', c=colors[int(data[0]) - 1])
#     pass


plt.show()
