# spots [![Build Status](https://travis-ci.org/lecardozo/spots.svg?branch=master)](https://travis-ci.org/lecardozo/spots)

Google Location History utilities

## Installation
```
$ pip install spots
```

## Usage
#### Load location history json as pandas DataFrame
```python
from spots import LocationHistory
locdf = LocationHistory.from_json("your-location-history-file.json")
locdf.head()

#   accuracy   activity                                         lat        lon               timestamp
#           confidence               timestamp     type                                              
#0       24        100 2014-01-05 09:47:07.808  UNKNOWN -23.340981 -46.579202 2014-01-05 09:47:07.808
#1       24        100 2014-01-05 09:47:54.558  TILTING -23.123471 -46.631244 2014-01-05 09:48:21.891
#2       24        100 2014-01-05 09:49:21.461  UNKNOWN -23.456211 -46.640234 2014-01-05 09:49:21.461
#3       24        100 2014-01-05 09:50:21.470  UNKNOWN -23.464231 -46.604355 2014-01-05 09:50:21.470
#4       25        100 2014-01-05 09:51:21.623  UNKNOWN -23.490080 -46.709021 2014-01-05 09:51:21.623

```

#### Calculate stay points for your trajectory
The `StayPointDetection` class implements the same interface used by `sklearn` clustering
algorithms.

```python
from spots import StayPointDetection
import numpy as np

spd = StayPointDetection(distance=0.05, time=np.timedelta(15, 'm'))
staypoints = spd.fit_predict(X=locdf[['lat', 'lon']].values, timestamp=locdf.timestamp)
locdf.loc[:, "staypoint_id"] = staypoints
```
