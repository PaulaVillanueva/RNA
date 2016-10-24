import numpy as np
import matplotlib.pyplot as plt
class HeatMap:
    def displayHeatMap (self, units, W, X, Y, aFn):
        color_matrix = np.zeros(units.shape)
        #Los colores son asignados directamente a las categorias si se representan con números contínuos

        #Idea: Para cada unidad los valores de activacion estan normalizados entre 0 y 1.
        #Para cada unidad sumo las activaciones discriminadas por categoria, sobre todos los samples
        #Al final voy a tener, para esa unidad, la cantidad de activacion total por categoria
        #Puedo normalizar ese vector y finalmente multiplico cada componente por su id de categoria correspondiente.

        for (i, j), value in np.ndenumerate(units):
            w = W(i,j)
            cat_buckets = np.zeros(Y.shape)

            for (x,y) in zip(X,Y):
                cat_buckets[y] += aFn(np.dot (np.transpose(x), w))

            cat_buckets_normalized = cat_buckets / np.linalg.norm(cat_buckets)

            final_color = 0
            for y in Y:
                final_color+= cat_buckets_normalized[y] * y

            color_matrix[i,j] = final_color / len(Y)

        plt.matshow(color_matrix)