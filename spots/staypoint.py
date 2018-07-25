import numpy as np
import pandas as pd
from datetime import timedelta
from haversine import haversine
from sklearn.base import BaseEstimator, ClusterMixin

class StayPointDetection(BaseEstimator, ClusterMixin):

    def __init__(self, time=timedelta(minutes=15), distance=0.05):
        self.time = np.timedelta64(time)
        self.distance = distance

    def fit(self, X, timestamps):
        """
        Find stay points based on time and distance threshold
        Implementation based on:
        Li et al. (2008). Mining user similarity based on location history
        """
        npoints = X.shape[0]
        stay_labels = np.full(npoints, -1)
        stay_id = 0
        i = 0
        while i < npoints:
            j = i + 1
            flag = False
            while j < npoints:
                center = [X[i,0], X[i,1]]
                dist = haversine(center, [X[j,0], X[j,1]])
                if dist > self.distance:
                    delta_t = timestamps[j] - timestamps[i]
                    if delta_t > self.time:
                        stay_labels[i:j] = stay_id
                        i = j
                        stay_id += 1
                        flag = True
                    break
                j += 1
            if not flag:
                i += 1

        self.labels_ = stay_labels

    def fit_predict(self, X, timestamps):
        self.fit(X, timestamps)
        return self.labels_
