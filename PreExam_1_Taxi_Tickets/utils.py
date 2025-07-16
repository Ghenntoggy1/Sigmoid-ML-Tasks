import pandas as pd
from dotenv import load_dotenv
import os
import requests
import osmnx as ox
import networkx as nx
import numpy as np
from haversine import haversine, Unit
import time
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from kydavra import PValueSelector

ox.settings.use_cache = False
ox.settings.log_console = False

load_dotenv()

DISTANCE_AI_API_KEY = os.getenv('DISTANCE_AI_API_KEY')

def get_percentage_cat_col(df: pd.DataFrame, col_name: str) -> pd.Series:
    return df[col_name].value_counts(normalize=True) * 100

def extract_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    datetime_features = ['pickup_datetime', 'dropoff_datetime']
    for col in datetime_features:
        df[col] = df[col].swifter.progress_bar(enable=True, desc=f'Convert {col} to DateTime').apply(pd.to_datetime, format='%Y-%m-%d %H:%M:%S')
        col_rounded = df[col].dt.round('H')
        df[f'{col.split(sep="_")[0]}_month'] = col_rounded.dt.month
        df[f'{col.split(sep="_")[0]}_day'] = col_rounded.dt.weekday
        df[f'{col.split(sep="_")[0]}_hour'] = col_rounded.dt.hour

        df[f'{col.split(sep="_")[0]}_period_of_day'] = df[f'{col.split(sep="_")[0]}_hour'].swifter.progress_bar(enable=True, desc=f'Convert {col} to DateTime').apply(
            lambda x: 'Morning' if 5 <= x < 12 else
                      'Afternoon' if 12 <= x < 17 else
                      'Evening' if 17 <= x < 22 else
                      'Night'
        )
        df[f'{col.split(sep="_")[0]}_is_weekend'] = df[f'{col.split(sep="_")[0]}_day'].swifter.progress_bar(enable=True, desc=f'Convert {col} to DateTime').apply(
            lambda x: 1 if x in [5, 6] else 0
        )
        df[f'{col.split(sep="_")[0]}_season'] = df[f'{col.split(sep="_")[0]}_month'].swifter.progress_bar(enable=True, desc=f'Convert {col} to DateTime').apply(
            lambda x: 'Winter' if x in [12, 1, 2] else
                      'Spring' if x in [3, 4, 5] else
                      'Summer' if x in [6, 7, 8] else
                      'Autumn'
        )

    return df

def extract_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    df['road_distance'] = df[['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']].swifter.progress_bar(enable=True, desc='Distance Calculation').apply(
        lambda rec: get_distance_haversine(row=rec), axis=1
    )
    return df

def get_distance_api(row: pd.Series) -> float:
    origin_lat = row['pickup_latitude']
    origin_long = row['pickup_longitude']
    destination_lat = row['dropoff_latitude']
    destination_long = row['dropoff_longitude']

    url = f"https://api-v2.distancematrix.ai/maps/api/distancematrix/json?origins={origin_lat},{origin_long}&destinations={destination_lat},{destination_long}&key={DISTANCE_AI_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        distance = int(data['rows'][0]['elements'][0]['distance']['value'])
        return distance
    else:
        raise Exception(f"Error fetching distance: {response.status_code} - {response.text}")

def get_distance_haversine(row: pd.Series) -> float:
    origin = (row['pickup_latitude'], row['pickup_longitude'])
    destination = (row['dropoff_latitude'], row['dropoff_longitude'])
    return haversine(origin, destination, unit=Unit.METERS)

def get_distance_osmnx(row: pd.Series, G: nx.Graph) -> float:
    try: 
        origin = (row['pickup_latitude'], row['pickup_longitude'])
        destination = (row['dropoff_latitude'], row['dropoff_longitude'])

        origin_node = ox.distance.nearest_nodes(G, X=origin[1], Y=origin[0])
        destination_node = ox.distance.nearest_nodes(G, X=destination[1], Y=destination[0])
        
        route_length_m = nx.shortest_path_length(G, origin_node, destination_node, weight='length')

        return route_length_m
    except Exception as e:
        print(f"Error calculating distance for row {row.name}: {e}")
        return np.nan

def measure_time_function(func, *args, **kwargs) -> tuple[float, float]:
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return result, elapsed_time

def get_outliers_by_boxplot(df: pd.DataFrame, col: str) -> pd.DataFrame:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

# class OutlierRemover(BaseEstimator, TransformerMixin):
#     def __init__(self, contamination=0.01, method='isolation_forest'):
#         self.contamination = contamination
#         self.method = method

#     def fit(self, X, y=None):
#         if self.method == 'isolation_forest':
#             self.model = IsolationForest(contamination=self.contamination, random_state=42)
#         elif self.method == 'local_outlier_factor':
#             self.model = LocalOutlierFactor(n_neighbors=20, contamination=self.contamination)
#         else:
#             raise ValueError(f"Unknown method: {self.method}")

#         self.model.fit(X)
#         self.inliers_ = self.model.predict(X) == 1
#         return self

#     def transform(self, X, y=None):
#         if hasattr(self, 'inliers_'):
#             return X[self.inliers_]
#         return X
    
class FeatureSelectorWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, selector):
        self.selector = selector
        self.selected_columns = None

    def fit(self, X, y):
        self.selected_columns = self.selector.select(pd.concat([X, y], axis=1), y_column=y.name)
        return self

    def transform(self, X):
        return X[self.selected_columns] if self.selected_columns is not None else X