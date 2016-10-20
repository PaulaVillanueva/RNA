import numpy as np
import matplotlib.pyplot as plt
class HeatMap:
    def displayHeatMap (self, units, W, X, Y, aFn):
        color_matrix = np.zeros(units.shape)
        #Los colores son asignados directamente a las categorias si se representan con números contínuos

        for (i, j), value in np.ndenumerate(units):
            w = W(i,j)
            cat_buckets = {}
            for y in Y:
                cat_buckets[y] = 0
            for (x,y) in zip(X,Y):
                cat_buckets[y] += aFn(np.dot (np.transpose(x), w))

            final_color = 0
            for y in Y:
                final_color+= cat_buckets[y] * y

            color_matrix[i,j] = final_color / len(Y)

        plt.matshow(color_matrix)