from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import ahocorasick
import geopandas
import numpy as np
import osmnx as ox
import pandas as pd
from mappymatch.matchers.lcss.lcss import LCSSMatcher
from mappymatch.maps.nx.nx_map import NxMap, NetworkType
from mappymatch.constructs.geofence import Geofence
from mappymatch.constructs.trace import Trace
from sklearn.base import TransformerMixin, BaseEstimator
from tqdm.auto import tqdm
from CaGeo.algorithms import BasicFeatures as bf
from CaGeo.algorithms import SegmentFeatures as sf

from streetmatching.algorithms import best_fitting
from streetmatching.algorithms.utils import multi_street_join, resample, rolling_avg, df_to_basic_feat, resample_df
from streetmatching.classes.mr_sax import MrSAX

from functools import reduce


class StreetMatching(TransformerMixin, BaseEstimator):
    def __init__(self,
                 alphabet_size=10,  #numero di simboli sax
                 sampling_distance=10,  #distanza, in metri, per il resample
                 smoothing=1,  #dimensione della sliding window per lo smoothing
                 street_padding=0,  #quanti samples eliminare dalla testa e dalla coda
                 sampling_accuracy=1,  #quante volte rieseguire l'operazione di resampling (> = >accuracy)
                 n_street_concat=1,  #Per ogni strada, quante altre strade concatenare
                 n_jobs=1,  #numero di workers paralleli
                 verbose=False  #verbosity
                 ):
        self.resampled_arches = None
        self.automaton = None
        self.sax = None
        self.arches = None

        self.alphabet_size = alphabet_size
        self.sampling_distance = sampling_distance
        self.smoothing = smoothing
        self.street_padding = street_padding
        self.sampling_accuracy = sampling_accuracy
        self.n_street_concat = n_street_concat
        self.n_jobs = n_jobs
        self.verbose = verbose


    def fit(self, trace, address=None, place=None, bbox=None, network_type="drive"):
        if address is not None:
            graph = ox.graph_from_address(address, network_type=network_type)
        elif place is not None:
            graph = ox.graph_from_place(place, network_type=network_type)
        elif bbox is not None:
            graph = ox.graph_from_bbox(*bbox, network_type=network_type)
        else:
            raise ValueError("At least one of address or place or bbox must be provided")

        resampled_trace = resample_df(trace, self.sampling_distance, self.sampling_accuracy)

        #sf.straightness(resampled_trace.X.values, resampled_trace.Y.values)

        self.arches = ox.graph_to_gdfs(graph, nodes=False).reset_index()
        self.arches = self.arches[["u", "v", "key", "oneway", "length", "geometry", "name"]]
        self.arches["name"] = self.arches["name"].apply(lambda x: f"{x}")
        self.arches.drop_duplicates(inplace=True)
        self.resampled_arches = self._parallel_multijoin(self.arches)
        self.resampled_arches = self._parallel_resampling()

        straightness = np.zeros(len(self.resampled_arches))
        for i, coords in enumerate(self.resampled_arches.geometry.values):
            coords = np.array(coords.xy)
            straightness[i] = sf.straightness(coords[0], coords[1])

        self.resampled_arches = self.resampled_arches[straightness < .9999] # gli altri sono inutili


        self.sax = MrSAX(self.alphabet_size, compress=1)\
            .fit(df_to_basic_feat(resampled_trace, bf.direction)[1:].reshape(1, -1))
        self.automaton = ahocorasick.Automaton()

        words = []

        for idx, el in enumerate(tqdm(self.resampled_arches.reset_index().to_dict("records"), disable=not self.verbose,
                                      desc="Building Automaton")):
            direction = rolling_avg(
                df_to_basic_feat(pd.DataFrame(el["geometry"].coords, columns=["X", "Y"]), bf.direction)[1:])
            # plt.plot(direction)
            street_name = el["name"] if type(el["name"]) == str else "None"

            word = "".join(map(str, self.sax.transform(direction.reshape(1, -1))[0]))
            if word not in self.automaton:
                assert self.automaton.add_word(word, ([el["key"]], [street_name]))
            else:
                old_values = self.automaton.pop(word)
                new_values = (old_values[0]+[el["key"]], old_values[1]+[street_name])
                assert self.automaton.add_word(word, new_values)
            # print(word)

            if el["oneway"]:
                words.append([word, np.nan])
            else:
                direction = rolling_avg(
                df_to_basic_feat(pd.DataFrame(list(el["geometry"].coords)[::-1], columns=["X", "Y"]), bf.direction)[1:])
                rev_word = "".join(map(str, self.sax.transform(direction.reshape(1, -1))[0]))

                if rev_word not in self.automaton:
                    assert self.automaton.add_word(rev_word, ([el["key"]], ["rev_"+street_name]))
                else:
                    old_values = self.automaton.pop(rev_word)
                    new_values = (old_values[0] + [el["key"]], old_values[1] + ["rev_"+street_name])
                    assert self.automaton.add_word(rev_word, new_values)

                words.append([word, rev_word])

        self.resampled_arches[["word", "rev_word"]] = words


        self.automaton.make_automaton()

        return self


    def _parallel_multijoin(self, arches):
        if self.n_jobs == 1:
            return multi_street_join(arches, self.n_street_concat, arches)

        executor = ProcessPoolExecutor(max_workers=self.n_jobs)
        multiprocess_dataframe_size = max(1, len(arches) // (100 * self.n_jobs))
        processes = []

        for i in range(0, len(arches), multiprocess_dataframe_size):
            processes += [
                executor.submit(multi_street_join, arches.iloc[i:i+multiprocess_dataframe_size],
                                                  self.n_street_concat,
                                                  arches)
            ]

        results = [x.result() for x in tqdm(processes, disable=not self.verbose, desc="Joining Streets")]
        results = [x for x in results if type(x) is not bool]

        executor.shutdown()

        return geopandas.GeoDataFrame(
            pd.concat(results, ignore_index=True),
            crs="EPSG:4326"
        )

    def _parallel_resampling(self):
        if self.n_jobs == 1:
            return resample(self.resampled_arches, self.sampling_distance, self.sampling_accuracy, self.street_padding)

        executor = ProcessPoolExecutor(max_workers=self.n_jobs)
        multiprocess_dataframe_size = max(1, len(self.resampled_arches) // (10 * self.n_jobs))
        processes = []
        for i in range(0, len(self.resampled_arches), multiprocess_dataframe_size):
            processes += [executor.submit(resample, self.resampled_arches.iloc[i:i + multiprocess_dataframe_size],
                                          self.sampling_distance, self.sampling_accuracy, self.street_padding)]

        executor.shutdown()

        return geopandas.GeoDataFrame(
            pd.concat([x.result() for x in tqdm(processes, disable=not self.verbose, desc="Resampling")],
                      ignore_index=True),
            crs="EPSG:4326"
        )

    def transform(self, trace:pd.DataFrame|list[pd.DataFrame]): # mi aspetto un dataframe con X e Y
        if type(trace) != pd.DataFrame:
            return [self.transform(x) for x in trace]

        resampled_trace = resample_df(trace, self.sampling_distance, self.sampling_accuracy)

        return resampled_trace, "".join(map(str, self.sax.transform(df_to_basic_feat(resampled_trace, bf.direction)[1:].reshape(1, -1))[0]))


    def search(self, trace:pd.DataFrame|list[pd.DataFrame]):
        if type(trace) != pd.DataFrame:
            return [self.search(x) for x in trace]

        trace = trace.copy()
        resampled_trace, trace_symbols = self.transform(trace)
        resampled_trace["symbol"] = ["None"]+[x for x in trace_symbols]

        possible_substreets_long = set(
            reduce(lambda x, y: x + y, [x[1][0] for x in self.automaton.iter_long(trace_symbols)]))
        possible_substreets = set(reduce(lambda x, y: x + y, [x[1][0] for x in self.automaton.iter(trace_symbols)]))

        print(f"Original search space={len(self.resampled_arches)}; "
              f"now searching only in {len(possible_substreets_long)} (worst case={len(possible_substreets)})"
              f" <=> {len(possible_substreets_long)/len(self.resampled_arches)*100}% and "
              f"{len(possible_substreets)/len(self.resampled_arches)*100}%")



        return self._search_fitting(resampled_trace, possible_substreets)

    def _search_fitting(self, resampled_trace, possible_substreets):
        res = []

        trace_xy = resampled_trace[["X", "Y"]].values

        for (u, v) in tqdm(possible_substreets):
            matched_street = self.resampled_arches[(self.resampled_arches.u == u) & (self.resampled_arches.v == v)]
            matched_street = matched_street.drop_duplicates()

            coords = np.array(matched_street.iloc[0].geometry.coords)

            if len(coords) < 10:
                continue

            alignment, score = best_fitting.euclidean_bestFit(coords, trace_xy)
            _, score_unnorm = best_fitting.euclidean_bestFit(coords, trace_xy, normalize=False)
            dist = bf.distance(coords.T[0], coords.T[1]).sum()
            print(score, matched_street.iloc[0]["name"])

            resampled_trace_shifted = resampled_trace[["X", "Y"]] - resampled_trace[["X", "Y"]].iloc[alignment] + coords[0]
            name = matched_street.iloc[0]["name"]

            mappy_score = _mappy_match(resampled_trace_shifted)

            remaning = resampled_trace_shifted.values[alignment + len(matched_street):]
            _score, _dist, _name = _depth_first_search(
                remaning,
                matched_street.iloc[0].v,
                bf.distance(remaning.T[0], remaning.T[1]).sum(),
                self.arches,
                sampling_distance=self.sampling_accuracy,
                sampling_accuracy=self.sampling_accuracy,
                street_padding=self.street_padding

            )
            score += _score
            dist += _dist
            name += " --A " + _name

















            res.append(((mappy_score, score, score_unnorm, dist, name),  matched_street.iloc[0]),)

            #print(alignment, score)

        print("\r\n\r\n")

        result_df = pd.DataFrame([x[1] for x in sorted(res, key=lambda x: -x[0][0])])

        result_df[["mappy_score", "distanza", "distanza_unnorm", "len_allineamento", "strade"]] = (
            pd.DataFrame([x[0] for x in sorted(res, key=lambda x: -x[0][1])],
                         columns=["mappy_score","distanza", "distanza_unnorm", "len_allineamento", "strade"])).values

        return result_df

def _mappy_match(my_trace):


    trace = Trace.from_dataframe(my_trace, lat_column="Y", lon_column="X")

    geofence = Geofence.from_trace(trace, padding=1e3)
    nx_map = NxMap.from_geofence(geofence, network_type=NetworkType.DRIVE)

    matcher = LCSSMatcher(nx_map)
    matched_points = matcher.match_trace(trace)

    matched_df = matched_points.matches_to_dataframe()
    matched_df = matched_df.groupby(by=["road_id"]).distance_to_road.mean()
    score = matched_df.mean()/len(matched_df)

    return score



def _depth_first_search(resampled_trace, v, max_depth=.5, arches=None,
                        sampling_distance=None,
                        sampling_accuracy=None,
                        street_padding=None,
                        reversed = False): #max_depth in km
    if len(resampled_trace) == 0 or max_depth < 0:
        return .0, .0, "consumata/depth"

    try:
        to_visit = pd.concat(
            [
                arches[arches.u == v],
                arches[(arches.v == v)]
            ]
        ).drop_duplicates().copy(deep=True)

        arches = arches[(arches.u != v) & ((arches.v != v) | (arches.oneway))]
    except Exception as e:
        print(e)

    if len(to_visit) == 0:
        return .0, .0, "no_exit"

    to_visit = resample(to_visit, sampling_distance, sampling_accuracy, street_padding)

    scores = np.zeros((len(to_visit)))
    matched = np.zeros(len(to_visit))
    names = []

    for i, row in enumerate(to_visit.copy(deep=True).to_dict("records")):
        coords = np.array(row["geometry"].coords)
        if len(resampled_trace[:min(len(coords)+2, len(resampled_trace))]) < len(coords):
            names.append("no_len")
            continue

        match_len = bf.distance(coords.T[0], coords.T[1]).sum()
        #print(round(match_len*1000, 100), "m")

        names.append(row["name"])
        idx, s = best_fitting.euclidean_bestFit(coords,
                                                resampled_trace[:min(len(coords) + 2, len(resampled_trace))],
                                                shift=True)

        rev_idx, rev_s = best_fitting.euclidean_bestFit(coords[::-1],
                                                        resampled_trace[:min(len(coords) + 2, len(resampled_trace))],
                                                        shift=True)

        s = min(s, rev_s)
        idx = [idx, rev_idx][np.argmin([s, rev_s])]


        score, dist, _name = _depth_first_search(
            resampled_trace[min(len(resampled_trace)-1, idx+len(coords)-1):],
            row["v"],
            max_depth-match_len,
            arches,
            sampling_distance, sampling_accuracy, street_padding
        )

        matched[i] = bf.distance(coords.T[0], coords.T[1]).sum() + dist
        scores[i] = s + score
        names[i] += " --> "+ _name

    return min(scores), matched[np.argmin(scores)], names[np.argmin(scores)]


# L'idea è di calcolare o score come euclidean_distance/street_length.
    #Se fatto a livello della funzione best_fitting, è semplice, ma l'aggregazione/media dei valori perde un po' di senso
    #Se fatto a livello della search (solo alla fine) penso che sia più reale il valore finale, ma non si può fare il pruning?
