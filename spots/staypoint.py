import numpy as np
import pandas as pd
from datetime import timedelta
from haversine import haversine


class StayPointDetection():

    def __init__(self, time=timedelta(minutes=15), distance=0.05):
        self.time = time
        self.distance = distance
        self.labels_ = None

    def fit(self, lochis):
        """
        Find stay points based on time and distance threshold
        Implementation based on:
        Li et al. (2008). Mining user similarity based on location history
        """
        df = lochis
        npoints = df.shape[0]
        stay_labels = np.full(npoints, -1)
        stay_id = 0
        i = 0
        while i < npoints:
            j = i + 1
            flag = False
            while j < npoints:
                center = df.loc[df.index[i], ['lat', 'long']]
                dist = haversine(center, df.loc[df.index[j], ['lat', 'long']])
                if dist > self.distance:
                    delta_t = df.loc[df.index[j], 'timestamp'] -\
                              df.loc[df.index[i], 'timestamp']
                    if delta_t[0] > self.time:
                        stay_labels[i:j] = stay_id
                        i = j
                        stay_id += 1
                        flag = True
                    break
                j += 1
            if not flag:
                i += 1

        self.labels_ = stay_labels

    def fit_predict(self, lochis):
        self.fit(lochis)
        return self.labels_
