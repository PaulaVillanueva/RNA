import numpy as np
import matplotlib.pyplot as plt

class FeatureNormalizer:

    def process(self, features):
        feature_means = np.mean(features, axis=0)
        features_std = np.std(features, axis=0)
        features_normalized = (features - feature_means) / features_std
        return features_normalized

class OutlierFilter:

    def process(self, features):
        features_std = np.std(features, axis=0)
        feature_means = np.mean(features, axis=0)
        features_wo_outliers = [item for item in features if self.isOutlier(item,feature_means,features_std)]
        print(len(features), len(features_wo_outliers))
        return features_wo_outliers

    def isOutlier(self, item, means, std):
        return (2*std + means - np.abs(item) > 0).all()
