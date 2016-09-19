import numpy as np

from preprocessor import OutlierFilter, FeatureNormalizer


class Ej1DataLoader:
    def LoadData(self):
        raw_data = np.genfromtxt('./ds/tp1_ej1_training.csv', delimiter=",")
        np.random.shuffle(raw_data)
        #features = raw_data[:, 1:]
        #labels = raw_data[:, 0]
        #transformed_features=OutlierFilter().process(features)
        #transformed_features=FeatureNormalizer().process(transformed_features)
        #return [transformed_features,labels]

        #Primero filtramos labels con features, luego separamos
        transformed_data=OutlierFilter().process(raw_data)
        transformed_data=FeatureNormalizer().process(transformed_data)
        features = transformed_data[:, 1:]
        labels = transformed_data[:, 0]
        labels = labels.reshape((labels.shape[0],1))
        #print("features ", features.shape)
        #print("labels shape ",labels.shape)
        return [features,labels]
