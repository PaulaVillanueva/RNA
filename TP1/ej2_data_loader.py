import numpy as np

from preprocessor import OutlierFilter, FeatureNormalizer


class Ej2DataLoader:
    def LoadData(self):
        raw_data = np.genfromtxt('/Users/bpanarello/Dropbox/RN/RNA/TP1/ds/tp1_ej2_training.csv', delimiter=",")
        np.random.shuffle(raw_data)

        #Primero filtramos labels con features, luego separamos
        transformed_data=OutlierFilter().process(raw_data)


        #transformed_data=FeatureNormalizer().process(transformed_data)
        features = transformed_data[:, 0:8]
        transformed_features = FeatureNormalizer().process(features)


        labels = transformed_data[:, 8:]
        transformed_labels = FeatureNormalizer().process(labels)

        transformed_labels = transformed_labels.reshape((transformed_labels.shape[0],2))
        return [transformed_features,transformed_labels]
