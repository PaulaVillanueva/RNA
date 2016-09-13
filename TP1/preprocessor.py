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
        features_wo_outliers = np.array(filter(self.isOutlier(feature_means,features_std), features))

        print(len(features), len(features_wo_outliers))
        #histograma de la 3era feature con cortes de outliers
        plt.hist(features[:,2],bins=np.max(features[:,2])-np.min(features[:,2]))
        plt.axvline(feature_means[2]-2*features_std[2], color='b', linestyle='dashed', linewidth=2)
        plt.axvline(feature_means[2]+2*features_std[2], color='b', linestyle='dashed', linewidth=2)
        plt.show()
        
        return features_wo_outliers

    def isOutlier(self, means, std):
        return lambda item: (np.abs(means - item) < 2*std).all()
