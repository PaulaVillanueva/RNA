import numpy as np
from cStringIO import StringIO

class DataLoader:
    def LoadData(self, fname):

        str_data=open (fname, "r").read()

        raw_data = np.genfromtxt(StringIO(str_data), delimiter=",")
        np.random.shuffle(raw_data)

        labels = raw_data[:, 0]
        features = raw_data[:, 1:]

        #centramos los features
        centrated_features = self.centralize(features)

        return (centrated_features,labels)


    # los datos tienen que estar centrados en el 0, no normalizados
    def centralize(self, v):
        means = np.mean(v, axis=0)
        return v - means
