import numpy as np
import pandas as pd
import shapely
import geopandas as geopd
from shapely import LineString, MultiLineString
from tqdm.auto import tqdm
import CaGeo.algorithms.BasicFeatures as bf


def street_join(street, df, crs="EPSG:4326"):
    if type(street.geometry.values[0]) == MultiLineString:
        return street

    df = df.copy()
    street_u = street.u.values[0]
    street_v = street.v.values[0]
    street_oneway = street.oneway.values[0]
    street_name = street.name.values[0] if type(street.name.values[0]) is str else "None"

    new_df = dict()

    # altra_strada -> mia_strada
    for el in df[df.v == street_u].to_dict("records"):
        el["v"] = street_v
        if (el["u"], el["v"]) in new_df:
            continue

        el["osmid"] = None
        el["oneway"] = el["oneway"] or street_oneway
        el["length"] = float(el["length"]) + float(street.length.values[0])
        if type(el["name"]) is str:
            el["name"] += " --> " + street_name
        else:
            el["name"] = "None --> " + street_name

        el["geometry"] = shapely.ops.linemerge(shapely.MultiLineString([el["geometry"], street.geometry.values[0]]))
        new_df[(el["u"], el["v"])] = el
        # print(el["oneway"])

    # mia_strada --> altra_strada
    """for el in df[df.u == street_v].to_dict("records"):
        el["u"] = street_u
        if (el["u"], el["v"]) in new_df:
            continue
        el["osmid"] = None
        el["oneway"] = el["oneway"] or street_oneway
        el["length"] = float(el["length"]) + float(street.length.values[0])
        if type(el["name"]) is str:
            el["name"] = street_name + " --> " + el["name"]
        else:
            el["name"] = street_name + " --> None"

        el["geometry"] = shapely.ops.linemerge(shapely.MultiLineString([street.geometry.values[0], el["geometry"]]))

        if type(el["geometry"]) is MultiLineString:
            continue

        new_df[(el["u"], el["v"])] = el"""

    if len(new_df) == 0:
        return street

    return geopd.GeoDataFrame(pd.DataFrame(new_df.values()), crs=crs)


def multi_street_join(archi, n_times=1, original_archi=None, verbose=False):
    import warnings
    warnings.filterwarnings("ignore")

    if n_times == 0:
        archi["key"] = archi[["u", "v"]].apply(tuple, axis=1)
        return archi[archi.geom_type == "LineString"] #se ci sono MultiLineString (rotatorie) non sono processabili
    if original_archi is None:
        original_archi = archi.copy(deep=True)
    archi_joined = []

    for i in tqdm(range(len(archi)), disable=not verbose):
        df_tmp = archi.iloc[i:i + 1]

        archi_joined.append(street_join(df_tmp, original_archi))

    try:
        new_archi = geopd.GeoDataFrame(pd.concat(archi_joined, ignore_index=True))
    except ValueError:
        return False

    return multi_street_join(new_archi, n_times - 1, original_archi)


def rolling_avg(data, window=3):
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')


def df_to_basic_feat(df: pd.DataFrame, fun):
    if not np.all(df.X < df.Y):
        print("here")
    return fun(df.X.values, df.Y.values)

def resample_df(df_to_resample=pd.DataFrame, distance_in_meters=1, iterazioni=1):
    dist = bf.distance(df_to_resample.X.values, df_to_resample.Y.values, accurate=True).sum() * 1000

    resampling_perc = np.linspace(0, 1, max(2, round(dist / distance_in_meters)))

    ls = LineString(df_to_resample[["X", "Y"]])

    resampled_traceV2 = []

    for point in ls.interpolate(resampling_perc, normalized=True):
        x, y = point.coords.xy
        resampled_traceV2.append((x[0], y[0]))

    if iterazioni == 1:
        return pd.DataFrame(resampled_traceV2, columns=["X", "Y"])
    else:
        return resample_df(pd.DataFrame(resampled_traceV2, columns=["X", "Y"]), distance_in_meters, iterazioni - 1)


def resample(archi: geopd.GeoDataFrame, sampling_distance, sampling_accuracy, street_padding, verbose=False):
    archi = archi.copy(deep=True)
    for i in tqdm(range(len(archi)), disable=not verbose):
        df_tmp = pd.DataFrame(archi.iloc[i].geometry.coords, columns=["X", "Y"])
        tmp = resample_df(df_tmp, sampling_distance, sampling_accuracy)
        if street_padding != 0 and len(tmp) > street_padding * 2 + 2:
            tmp = tmp.iloc[street_padding:-street_padding:]
        archi.iloc[i, archi.columns.tolist().index("geometry")] = LineString(tmp)

    return archi