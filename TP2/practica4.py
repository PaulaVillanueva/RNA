import numpy as np
import matplotlib.pyplot as plt
from us_learning import HebbianNN

# los datos tienen que estar centrados en 00, no normalizados
# en este caso, se supone que ya estarian centrados por la uniforme alrededor del 0

def centralize(v):
    means = np.mean(v, axis=0)
    return v - means

# train

A = np.random.random_integers(1, 100, 6)
# A[0] = A[1]
print("A:", A)

epochs = 1000
eps = 0.0001
lrcons = 0.0001

DS = []
for t in range(epochs):
    X = [np.random.uniform(-a, a) for a in A]
    DS.append(X)

DS = np.array(DS, dtype=float)
DS = centralize(DS)

HB = HebbianNN(6, 4, 0.5)

# OjaM

we = HB.train(DS, eps, lrcons, epochs, True)
print we
we = HB.train_opt(DS, eps, lrcons, epochs, True)
print we

print "check orthogonality: ", HB.orthogonal(we,eps)
print "check norm == 1: ", np.linalg.norm(we, axis=0)

plt.matshow(we, cmap='hot', vmin=-1, vmax=1)
plt.colorbar()
plt.show()

outputs = np.array([np.dot(x.transpose(), we) for x in DS])

print "outputs: "
print "mean: ", np.mean(outputs, axis=0)
print "std: ", np.std(outputs, axis=0)
print "var: ", np.var(outputs, axis=0)

# Sanger

# eps = 0.007

# we = HB.train(DS, eps, lrcons, epochs)

# print we
# #print "check orthogonality: ", HB.orthogonal(we,eps)
# #print "check norm == 1: ", np.linalg.norm(we, axis=0)

# we = HB.train_opt(DS, eps, lrcons, epochs)
# print we
# #plt.matshow(we, cmap='hot', vmin=-1, vmax=1)
# #plt.colorbar()
# #plt.show()

# outputs = np.array([np.dot(x.transpose(), we) for x in DS])

# print("mean: ", np.mean(outputs, axis=0))
# print("std: ", np.std(outputs, axis=0))
# print("var: ", np.var(outputs, axis=0))

# # plt.plot(outputs[:,0], outputs[:,1], 'ro')
# # plt.show()

# # plt.plot(outputs[:,1], outputs[:,2], 'ro')
# # plt.show()

# # plt.plot(outputs[:,2], outputs[:,3], 'ro')
# # plt.show()