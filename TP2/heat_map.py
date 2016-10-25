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

        for i in range (layout[0]):
            for j in range (layout[1]):
                w = W[(i,j)]
                cat_buckets = np.zeros(Y.shape)

                for (x,y) in zip(X,Y):
                    cat_buckets[y] += aFn(np.dot (np.transpose(x), w))

                cat_buckets_normalized = cat_buckets / np.linalg.norm(cat_buckets)

                final_color = 0
                for y in Y:
                    final_color+= cat_buckets_normalized[y] * y

                color_matrix[i,j] = final_color / len(Y)

        plt.matshow(color_matrix)
        plt.show()
        raw_input("")