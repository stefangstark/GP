import os
import sys
import numpy as np
import numpy.linalg as la
import scipy.spatial.distance as ssd

def sqe(X, Y=None, l=1):
    if Y is None:
        pairwise_dists = ssd.squareform(ssd.pdist(X,'euclidean'))
    else:
        pairwise_dists = ssd.cdist(X, Y, 'euclidean')
    K = np.exp(-pairwise_dists ** 2 / (2*l))
    return np.matrix(K)

class KernelGP(object):
    def __init__(self, kernel, verbose=False):
        self.kernel = kernel
        self._observed = False
        self._v = verbose
        if self._v: print("KernelGP(%s kernel)\n" %kernel.__name__)
        return
    
    def observe(self, x, y, sigma):
        self.xobs = np.matrix(x)
        self.yobs = np.matrix(y)
        self.sigma = sigma
        self.d, self.n = self.xobs.shape
        if self._v: print("Oberved %d data points of dimension %d\n" %(self.n, self.d))

        K = np.matrix(self.kernel(self.xobs.T)) + sigma * np.identity(self.n)
        self.L = np.linalg.cholesky(K)
        assert(np.allclose(self.L * self.L.T, K))
        self.alpha = la.solve(self.L.T, la.solve(self.L, self.yobs.T))
        self._observed = True
        return

    def predict(self, x):
        assert(self._observed)
        x = np.matrix(x)
        k = self.kernel(self.xobs.T, x.T)
        predm = k.T * self.alpha
        v = la.solve(self.L, k)
        predv = sqe(x.T, x.T) - v.T * v 
        predm = np.squeeze(np.array(predm))
        predv = np.array(predv)
        return predm, predv

