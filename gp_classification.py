import os
import sys
import numpy as np
import numpy.linalg as la

from kernels import sqe

class GPLaplace(object):
    def __init__(self, l=1, verbose=False):
        self.l = l
        self._observed = False
        self._v = verbose
        self._lp_min_iter = 5
        self._lp_max_iter = 50
        return

    def kernel(self, X, Y=None):
        return sqe(X, Y, l=self.l)

    def observe(self, x, y):
        self.n, self.d = x.shape
        self.x = x
        self.y = y
        if self._v: print("Oberved %d data points of dimension %d\n" %(self.n, self.d))

        self._opt_laplace_approx()
        return

    def _opt_laplace_approx(self):
        self.K = self.kernel(self.x, self.x)
        self._obj_history = list()
        self.qmean = np.random.randn(self.n)
        pi = 1 / (1 + np.exp(-self.qmean))
        t = ( self.y + 1 ) / 2

        grad_mag= np.inf
        convg = np.allclose(grad_mag, 0)
        counter = 0
        while (counter < self._lp_min_iter or (counter < self._lp_max_iter and not convg )):
            Wsqrt = np.sqrt(pi * (1 - pi))
            B = np.identity(self.n) + Wsqrt[:, np.newaxis] * self.K * Wsqrt
            L = la.cholesky(B)
            b = Wsqrt * Wsqrt * self.qmean + t - pi
            Wkb = Wsqrt * np.matmul(self.K, b)
            a = b - Wsqrt * la.solve(L.T, la.solve(L, Wkb))

            self.qmean = np.matmul(self.K.T, a)
            pi = 1 / (1 + np.exp(-self.qmean))
            grad_mag = np.sum((t - pi - a) ** 2)
            self.log_q = -np.matmul(self.qmean.T, a) / 2 - \
                         np.sum(np.log(1 + np.exp(-self.y * self.qmean))) - \
                         np.sum(np.log(np.diagonal(L)))

            self._obj_history.append(self.log_q + np.sum(np.log(np.diagonal(L))))
            counter = counter + 1
            convg = np.allclose(grad_mag, 0)

        self._L = L
        self._pi = pi
        self._Wsqrt = Wsqrt
        return

    def predict(self, xstar, nsamp=20):
        k = self.kernel([xstar], self.x).T
        t = (self.y + 1) / 2
        predm = np.matmul(k.T, t - self._pi)
        v = np.squeeze(la.solve(self._L, self._Wsqrt[:, np.newaxis]*k))
        predv = np.squeeze(np.array(self.kernel([xstar], [xstar]))) - np.inner(v,v)
        assert(predv >= 0)

        z = np.random.randn(nsamp) * predv + predm
        predc = np.mean(1 / (1 + np.exp(-z)))
        return predc, predm, predv

