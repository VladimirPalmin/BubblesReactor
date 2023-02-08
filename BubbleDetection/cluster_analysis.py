from sklearn.cluster import DBSCAN
import numpy as np


def get_clusters(X):
    clustering = DBSCAN(eps=3, min_samples=5)
    labels = clustering.fit_predict(X)
    clusters = np.unique(labels)
    clusters = clusters[clusters>=0]
    return clusters, labels
