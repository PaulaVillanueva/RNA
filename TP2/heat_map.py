# -*- coding: latin-1 -*-
import numpy as np
import matplotlib.pyplot as plt
class HeatMap:
    def displayHeatMap (self, layout, W, X, Y, aFn):
        color_matrix = np.zeros((layout[0],layout[1]))
        #Los colores son asignados directamente a las categorias si se representan con numeros contínuos

        #Idea: Para cada unidad los valores de activacion estan normalizados entre 0 y 1.
        #Para cada unidad sumo las activaciones discriminadas por categoria, sobre todos los samples
        #Al final voy a tener, para esa unidad, la cantidad de activacion total por categoria
        #Puedo normalizar ese vector y finalmente multiplico cada componente por su id de categoria correspondiente.

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


        plt.matshow(final_colors)
        plt.show()
        raw_input("")