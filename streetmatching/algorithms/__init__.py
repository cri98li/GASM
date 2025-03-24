import numpy as np
import pandas as pd
from shapely import LineString
from CaGeo.algorithms import BasicFeatures as bf


def rolling_avg(data, window=3):
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='same')


def resample_df(df_to_resample=pd.DataFrame, distance_in_meters=1, iterazioni=2):
    dist = bf.distance(df_to_resample.x, df_to_resample.y, accurate=True).sum() * 1000

    resampling_perc = np.linspace(0, 1, max(2, round(dist / distance_in_meters)))

    ls = LineString(df_to_resample)

    resampled_traceV2 = []

    for point in ls.interpolate(resampling_perc, normalized=True):
        x, y = point.coords.xy
        resampled_traceV2.append((x[0], y[0]))

    if iterazioni == 1:
        return pd.DataFrame(resampled_traceV2, columns=["x", "y"])
    else:
        return resample_df(pd.DataFrame(resampled_traceV2, columns=["x", "y"]), distance_in_meters, iterazioni - 1)

