import numpy as np
import matplotlib.pyplot as plt
from us_learning import HebbianNN


def normalize(v):
    means = np.mean(v, axis=0)
    stds = np.std(v, axis=0)
    return (v - means) / stds


# train

A = np.random.random_integers(1, 100, 6)
print("A:", A)

epochs = 200

DS = []
for t in range(epochs):
    X = [np.random.uniform(-a, a) for a in A]
    DS.append(X)

DS = np.array(DS, dtype=float)
DS = normalize(DS)

HB = HebbianNN(6, 4)

# OjaM

we = HB.train(DS, 0.001, epochs, True)

print("weights OjaM: ")
print(we)
print("check orthogonality: ", HB.orthogonal(we,0.001))
plt.matshow(we, cmap='hot', vmin=-1, vmax=1)
plt.colorbar()
plt.show()

outputs = [np.dot(x.transpose(), we) for x in DS]

print("mean: ", np.mean(outputs, axis=0))
print("std: ", np.std(outputs, axis=0))
print("var: ", np.var(outputs, axis=0))

# Sanger

we = HB.train(DS, 0.001, epochs)

print("weights Sanger: ")
print(we)
print("check orthogonality: ", HB.orthogonal(we,0.001))
plt.matshow(we, cmap='hot', vmin=-1, vmax=1)
plt.colorbar()
plt.show()

outputs = [np.dot(x.transpose(), we) for x in DS]

print("mean: ", np.mean(outputs, axis=0))
print("std: ", np.std(outputs, axis=0))
print("var: ", np.var(outputs, axis=0))
