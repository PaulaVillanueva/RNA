import numpy as np

from preprocessor import OutlierFilter, FeatureNormalizer


class Ej2DataLoader:
    def LoadData(self, fname):
        raw_data = np.genfromtxt(fname, delimiter=",")
        np.random.shuffle(raw_data)

        #Primero filtramos labels con features, luego separamos
        transformed_data=OutlierFilter().process(raw_data)


        #transformed_data=FeatureNormalizer().process(transformed_data)
        features = transformed_data[:, 0:8]
        transformed_features = FeatureNormalizer().process(features,  useMaxMin=True)


        labels = transformed_data[:, 8:]
        transformed_labels = FeatureNormalizer().process(labels,  useMaxMin=True)

        transformed_labels = transformed_labels.reshape((transformed_labels.shape[0],2))
        return [transformed_features,transformed_labels]
