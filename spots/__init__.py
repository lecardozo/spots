import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
from datetime import timedelta
from glob import glob

from .staypoint import StayPointDetection

class LocationHistory:
    """
       Methods for loading and parsing
       Google's Location History json files.
    """

    @classmethod
    def from_json(cls, filename, *args, **kwargs):
        """
            LocationHistory DataFrame from json.

            Parameters
            ----------
            filename: Location history file (.json) name.

            Returns
            -------
            df: pandas.DataFrame
                Location history in tabular format.
        """
        with open(filename) as f:
            loc = json.load(f)['locations']
            for record in loc:
                if not record.get('activity'):
                    record['activity'] = [
                        {
                            'activity': [
                                {'type': 'UNKNOWN', 'confidence': 100}
                            ],
                            'timestampMs': record['timestampMs']
                        }
                    ]
                if not record.get('accuracy'):
                    record['accuracy'] = np.nan

        df = json_normalize(loc, ['activity', 'activity'],
                            ['timestampMs', 'latitudeE7', 'longitudeE7',
                             'accuracy', ['activity', 'timestampMs']],
                            record_prefix='activity.')
        try:
            process_activities = kwargs.pop('process_activities')
        except KeyError:
            process_activities = True

        return cls._preprocess(df, process_activities=process_activities)


    def _preprocess(df, process_activities=True):
        """
            Data preprocessing

            Returns
            -------
            df: pandas.DataFrame
                Preprocessed location history dataframe.
        """
        df.latitudeE7 = df.latitudeE7 * 10e-8
        df.longitudeE7 = df.longitudeE7 * 10e-8
        df.timestampMs = pd.to_datetime(df.timestampMs, unit='ms')
        df['activity.timestamp'] = pd.to_datetime(
            df['activity.timestampMs'], unit='ms'
        )
        df = df.drop(columns='activity.timestampMs')
        df.rename({'latitudeE7': 'lat', 'longitudeE7': 'lon',
                   'timestampMs': 'timestamp'},
                  axis='columns', inplace=True)

        if process_activities:
            df = df[(df.groupby('activity.timestamp')['activity.confidence']
                       .transform(max)) == df['activity.confidence']]
            df = df[(df.groupby('timestamp')['activity.confidence']
                       .transform(max)) == df['activity.confidence']]
            df = df.drop_duplicates(['activity.type',
                                     'activity.confidence',
                                     'timestamp'])
            df['activity.timedelta'] = np.abs(df['timestamp'] - df['activity.timestamp'])
            df = df[(df.groupby('timestamp')['activity.timedelta']
                       .transform(min)) == df['activity.timedelta']]
            df.drop(columns=['activity.timedelta'], inplace=True)

        df = df[sorted(df.columns)]
        cols = []
        for col in df.columns:
            if 'activity' in col:
                cols.append(col.split('.'))
            else:
                cols.append((col, ''))
        cols = pd.MultiIndex.from_tuples(cols)
        df.columns = cols
        df = df.sort_values('timestamp', ascending=True)
        df = df.reset_index(drop=True)
        return df
