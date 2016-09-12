import numpy as np


class FeatureNormalizer:

    def normalize_features(self, features):
        feature_means = np.mean(features, axis=0)
        features_std = np.std(features, axis=0)
        features_normalized = (features - feature_means) / features_std
        return features_normalized
