import pandas as pd

from streetmatching.classes.street_matching import StreetMatching

if __name__ == '__main__':
    traces = pd.read_csv("tracks_1/track_points.csv")
    print(f"# Traces: {len(traces.track_fid.unique())}\t{traces.track_fid.unique()}")

    #funziona bene con: 2, 1
    my_trace = traces[traces.track_fid == 2]

    sm = StreetMatching(
        sampling_distance=5,
        alphabet_size=8,
        street_padding=1,
        n_street_concat=2, #2
        smoothing=3,
        verbose=True,
        n_jobs=9
    )

    sm.fit(my_trace,
           address="Salviano, Livorno, Italy",
           #place="Livorno, Italy"
           )

    print(sm.transform(my_trace))

    print(sm.search(my_trace))