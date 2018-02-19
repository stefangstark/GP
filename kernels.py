import numpy as np
import scipy.spatial.distance as ssd

def sqe(X, Y=None, l=1):
    if Y is None:
        pairwise_dists = ssd.squareform(ssd.pdist(X,'euclidean'))
    else:
        pairwise_dists = ssd.cdist(X, Y, 'euclidean')
    K = np.exp(-pairwise_dists ** 2 / (2*l))
    return K

