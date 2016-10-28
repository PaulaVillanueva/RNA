from kohonen_category_mapper import KohonenCategoryMapper
import numpy as np

class KohonenClassifier:
    def __init__(self, layout, W, category_map ):
        self._layout = layout
        self._weights = W
        # Saco las unidades que no tienen categoria
        self._category_map = {k:v for k,v in category_map.iteritems() if v > 0}


    def classify(self, sample):

        # De las unidades con categoria, la que mas se acerca
        winner = min(self._weights.keys(), key=lambda k: np.linalg.norm(self._weights[k] - sample))
        return self._category_map[winner]





