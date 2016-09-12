import numpy as np

from preprocessor import OutlierFilter, FeatureNormalizer


class Ej1DataLoader:
    def LoadData(self):
        rawdata = np.genfromtxt('./ds/tp1_ej1_training.csv', delimiter=",")
        features = rawdata[:, 1:]
        labels = rawdata[:, 0]
        transformed_features=OutlierFilter().process(features)
        transformed_features=FeatureNormalizer().process(transformed_features)
        return [transformed_features,labels]
