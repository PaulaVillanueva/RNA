
import numpy as np
import matplotlib.pyplot as plt
class HeatMap:
    def __init__(self):
        self._figure = 0
        self._plot_handler = None

    def displayHeatMap (self, layout, W, X, Y, textOnly=True):
        color_matrix = np.zeros((layout[0],layout[1]))
        cat_qtys = {}
        for i in range (layout[0]):
            for j in range (layout[1]):
                cat_qtys[(i,j)] = np.zeros(10)

        for (x, y) in zip(X, Y):
            winner = min(W.keys(), key=lambda k: np.linalg.norm(W[k] - x))
            cat_qtys[winner][y] = cat_qtys[winner][y] + 1


        final_colors = np.zeros(layout)
        for i in range (layout[0]):
            for j in range (layout[1]):
                final_colors[i][j] = np.argmax(cat_qtys[(i,j)])

        if textOnly:
            print (final_colors)
        else:
            plt.ion()
            plt.matshow(final_colors)
            plt.show()

    def show_from_category_dictionary(self, cdict, layout, textOnly=True):
        final_colors = np.zeros(layout)
        for i in range (layout[0]):
            for j in range (layout[1]):
                final_colors[i][j] = cdict[(i,j)]

        print final_colors
        if not textOnly:
            plt.ion()
            plt.matshow(final_colors)
            plt.show()
