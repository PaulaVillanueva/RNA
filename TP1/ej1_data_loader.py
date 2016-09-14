import numpy as np

from preprocessor import OutlierFilter, FeatureNormalizer


class Ej1DataLoader:
    def LoadData(self):
        raw_data = np.genfromtxt('./ds/tp1_ej1_training.csv', delimiter=",")
        np.random.shuffle(raw_data)
        features = raw_data[:, 1:]
        labels = raw_data[:, 0]
        transformed_features=OutlierFilter().process(features)
        transformed_features=FeatureNormalizer().process(transformed_features)
        return [transformed_features,labels]
