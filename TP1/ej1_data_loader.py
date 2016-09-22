import numpy as np

from preprocessor import OutlierFilter, FeatureNormalizer


class Ej1DataLoader:
    def LoadData(self):
        raw_data = np.genfromtxt('./ds/tp1_ej1_training.csv', delimiter=",")
        #np.random.shuffle(raw_data)

        #Primero filtramos labels con features, luego separamos
        transformed_data=OutlierFilter().process(raw_data)

        features = transformed_data[:, 1:]
        labels = transformed_data[:, 0]
        labels = labels.reshape((labels.shape[0],1))

        #Normalizamos features
        transformed_features = FeatureNormalizer().process(features)

        return [transformed_features,labels]
