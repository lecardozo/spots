import numpy as np
from numba import jit, f8, i8
from numba.types import NPDatetime, NPTimedelta
from sklearn.base import BaseEstimator, ClusterMixin


@jit(f8(f8[:], f8[:]), nopython=True)
def haversine(a, b):
    """ Calculate haversine distance

    Parameters
    ----------
    a, b: np.ndarray
        One-dimensional np.ndarray with latitude (a[0]) and
        longitude (a[1]) values in degrees

    Return
    ------
    dist: np.float64
        Distance between `a` and `b` in kilometers
    """
    a = np.radians(a)
    b = np.radians(b)
    delta_lat = b[0] - a[0]
    delta_lon = a[0] - a[1]
    dist = (np.sin(delta_lat * 0.5) ** 2 + np.cos(a[0]) *
            np.cos(b[0]) * np.sin(delta_lon * 0.5) ** 2)
    dist = 2 * 6371 * np.arcsin(np.sqrt(dist))

    return dist


@jit(i8[:](f8[:, :], NPDatetime('ns')[:], f8, NPTimedelta('ns')),
     nopython=True)
def _detect_staypoints(X, timestamps, distance, time):
    """
    Find stay points based on time and distance threshold.

    Parameters
    ----------
    X: numpy.ndarray
        Two-dimensional array for which each row represents
        a latitude-longitude pair. Pairs must be chronologically
        sorted (from the oldest to the newest).
    timestamps: numpy.ndarray
        One-dimensional array of timestamps relative to X points.
        timestamps.shape[0] must be equal to X.shape[0].
    distance: float
        Distance threshold used to define a stay point.
    time: numpy.timedelta64
        Time threshold used to define a stay point.

    Returns
    -------
    stay_labels: numpy.ndarray
        Cluster lables for each point. Non-stay points are given
        the label -1.

    References
    ----------
    Li et al., "Mining user similarity based on location history".
    In: Proceedings of the 16th ACM SIGSPATIAL international
    conference on Advances in geographic information systems. 2008
    """
    npoints = X.shape[0]
    stay_labels = np.full(npoints, -1, dtype=np.int64)
    stay_id = 0
    i = 0
    while i < npoints:
        j = i + 1
        flag = False
        while j < npoints:
            center = np.array([X[i, 0], X[i, 1]])
            next_point = np.array([X[j, 0], X[j, 1]])
            dist = haversine(center, next_point)
            if dist > distance:
                delta_t = timestamps[j] - timestamps[i]
                if delta_t > time:
                    stay_labels[i:j] = stay_id
                    i = j
                    stay_id += 1
                    flag = True
                break
            j += 1
        if not flag:
            i += 1
    return stay_labels


class StayPointDetection(BaseEstimator, ClusterMixin):
    """
    Find stay points based on time and distance threshold.

    Parameters
    ----------
    X: numpy.ndarray
        Two-dimensional array for which each row represents
        a latitude-longitude pair. Pairs must be chronologically
        sorted (from the oldest to the newest).
    timestamps: numpy.ndarray
        One-dimensional array of timestamps relative to X points.
        timestamps.shape[0] must be equal to X.shape[0].
    distance: float
        Distance threshold used to define a stay point.
    time: numpy.timedelta64
        Time threshold used to define a stay point.

    Attributes
    -------
    labels_: array, shape = X.shape[0]
        Cluster labels for each point. Non-stay points are given
        the label -1.

    References
    ----------
    Li et al., "Mining user similarity based on location history".
    In: Proceedings of the 16th ACM SIGSPATIAL international
    conference on Advances in geographic information systems. 2008
    """

    def __init__(self, distance=0.05,
                 time=np.timedelta64(15, 'm')):
        self.time = time
        self.distance = distance

    def fit(self, X, timestamps):
        """ Performs stay point detection """
        self.labels_ = _detect_staypoints(X, timestamps,
                                          self.distance,
                                          self.time.astype('timedelta64[ns]'))

    def fit_predict(self, X, timestamps):
        """ Performs stay point detection and returns stay points labels """
        self.fit(X, timestamps)
        return self.labels_
