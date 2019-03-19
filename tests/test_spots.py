import pytest
import numpy as np
from spots import LocationHistory, StayPointDetection


@pytest.fixture(scope='session')
def data():
    return '''
    {
      "locations" : [ {
        "timestampMs" : "1511361563993",
        "latitudeE7" : -31367382,
        "longitudeE7" : -702350293,
        "accuracy" : 64
      }, {
        "timestampMs" : "1511361512903",
        "latitudeE7" : -40887882,
        "longitudeE7" : -703350253,
        "accuracy" : 28,
        "altitude" : 32,
        "activity" : [ {
          "timestampMs" : "1511361512897",
          "activity" : [ {
            "type" : "UNKNOWN",
            "confidence" : 31
          }, {
            "type" : "STILL",
            "confidence" : 17
          }, {
            "type" : "ON_FOOT",
            "confidence" : 14
          }, {
            "type" : "WALKING",
            "confidence" : 14
          }, {
            "type" : "IN_VEHICLE",
            "confidence" : 12
          }, {
            "type" : "IN_ROAD_VEHICLE",
            "confidence" : 12
          }, {
            "type" : "IN_RAIL_VEHICLE",
            "confidence" : 11
          }, {
            "type" : "ON_BICYCLE",
            "confidence" : 8
          }, {
            "type" : "RUNNING",
            "confidence" : 8
          } ]
        } ]
      }
    ] }
    '''


@pytest.fixture()
def location_json(tmpdir_factory, data):
    p = tmpdir_factory.mktemp('data').join('location.json')
    p.write(data)
    return str(p)


@pytest.fixture()
def default_columns():
    default_cols = [('accuracy', ''),
                    ('activity', 'confidence'),
                    ('activity', 'timestamp'),
                    ('activity', 'type'),
                    ('lat', ''),
                    ('lon', ''),
                    ('timestamp', '')]
    return set(default_cols)


@pytest.fixture()
def lhistory(location_json):
    return LocationHistory.from_json(location_json)


def test_columns(lhistory, default_columns):
    assert set(lhistory.columns) == default_columns


def test_stay_point_detection(lhistory):
    spd = StayPointDetection()
    labels = spd.fit_predict(lhistory[['lat', 'lon']].values,
                             lhistory['timestamp'].values)
    assert isinstance(labels, np.ndarray)
