import numpy as np
import matplotlib.pyplot as plt

class FeatureNormalizer:

    def process(self, features):
        """
        :param features: datos a normalizar
        :return: devuelve datos normalizados por columna
        """
        feature_means = np.mean(features, axis=0)
        features_std = np.std(features, axis=0)
        features_normalized = (features - feature_means) / features_std
        return features_normalized

class OutlierFilter:

    def process(self, features):
        """
        :param features: datos
        :return: datos sin outliers
        """
        features_std = np.std(features, axis=0)
        feature_means = np.mean(features, axis=0)

        features_wo_outliers = np.array(filter(self.isOutlier(feature_means,features_std), features))

        #print(len(features), len(features_wo_outliers))

        #for i in range(len(features[0])):
            ##boxplot
            #fig = plt.figure(1, figsize=(9, 6))
            #ax = fig.add_subplot(111)
            #bp = ax.boxplot([features[:,i],features_wo_outliers[:,i]])
            #plt.show()

            ##histograma de la 3era feature con cortes de outliers
            #plt.hist(features[:,i],bins=np.max(features[:,i])-np.min(features[:,i]))
            #plt.axvline(feature_means[i]-2*features_std[i], color='b', linestyle='dashed', linewidth=2)
            #plt.axvline(feature_means[i]+2*features_std[i], color='b', linestyle='dashed', linewidth=2)
            #plt.show()
        
        #print("features_wo_outliers:", features_wo_outliers.shape)

        return features_wo_outliers

    def isOutlier(self, means, std):
        return lambda item: (np.abs(means - item) < 2*std).all()
