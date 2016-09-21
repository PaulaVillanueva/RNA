import matplotlib.pyplot as plt
import numpy as np


def plot_error(errors, y_title):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    num_epochs = len(errors)
    ax.plot(np.arange(0, num_epochs),
            errors[0:num_epochs])
    ax.set_xlim([0, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title(y_title)
    plt.show()
