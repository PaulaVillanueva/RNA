# -*- coding: latin-1 -*-
import numpy as np
import matplotlib.pyplot as plt

class KohonenCategoryMapper:
    def getCategoryMap (self, layout, W, X, Y):
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

        return final_colors