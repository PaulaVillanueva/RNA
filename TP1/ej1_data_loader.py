import numpy as np
from cStringIO import StringIO
from preprocessor import OutlierFilter, FeatureNormalizer


class Ej1DataLoader:
    def LoadData(self, fname):


        str_data=open (fname, "r").read()

        #Reemplazo M y B por enteros
        str_data = str_data.replace("B", "0")
        str_data = str_data.replace("M", "1")

        raw_data = np.genfromtxt(StringIO(str_data), delimiter=",")
        #np.random.shuffle(raw_data)

        #Primero filtramos labels con features, luego separamos
        transformed_data=OutlierFilter().process(raw_data)

        features = transformed_data[:, 1:]
        labels = transformed_data[:, 0]
        labels = labels.reshape((labels.shape[0],1))

        #Normalizamos features
        transformed_features = FeatureNormalizer().process(features)

        return [transformed_features,labels]
