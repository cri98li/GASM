import numpy as np
from CaGeo.algorithms import BasicFeatures as bf

def euclidean(sub:np.array, ts:np.array)->float:
    dist = (sub - ts) ** 2
    return np.sqrt(dist.sum())

def euclidean_bestFit(sub:np.array, ts:np.array, shift=True, normalize=True) ->float:
    if len(sub) > len(ts):
        return (-1, np.inf)

    if (sub - sub[0]).sum() == .0:
        return (-1, np.inf)

    best_alignment = (-1, np.inf)

    for i in range(len(ts)-len(sub)+1):
        sub_shifted = sub
        if shift:
            sub_shifted = sub-sub[0]+ts[i]
        dist = euclidean(sub_shifted, ts[i:i+len(sub)])

        if dist < best_alignment[1]:
            best_alignment = (i, dist)

    if normalize:
        #best_alignment = (best_alignment[0], best_alignment[1]/bf.distance(sub.T[0], sub.T[1], accurate=False).sum())
        best_alignment = (best_alignment[0], best_alignment[1] / len(sub[0]))

    return best_alignment

if __name__ == '__main__':
    sub = np.array([
        [1, 2, 3],
        [1,2,3]
    ]).T

    ts = np.array([
        [3,2,1,2,3,9,3,4],
        [13, 12, 11, 12, 13, 19, 13, 14]
    ]).T

    idx, val = euclidean_bestFit(sub, ts)

    print(idx, val)
    print(sub)
    print(ts[idx:idx+len(sub)])