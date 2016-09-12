import numpy as np


class FeatureNormalizer:

    def process(self, features):
        feature_means = np.mean(features, axis=0)
        features_std = np.std(features, axis=0)
        features_normalized = (features - feature_means) / features_std
        return features_normalized

class OutlierFilter:
    def process(self, features):
        return features

