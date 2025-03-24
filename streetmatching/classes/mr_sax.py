import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator


class MrSAX(TransformerMixin, BaseEstimator):

    def __init__(self, n_bins, strategy="custom", compress=1):
        self.n_bins = n_bins
        self.strategy = strategy
        self.compress = compress
        self.symbols = "abcdefghijklmnopqrstuvwxyz"+"0123456789"+"ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        if n_bins > len(self.symbols):
            raise ValueError(f"n_bins must be at most {len(self.symbols)}")

    def fit(self, X):
        self.thr = np.linspace(-np.pi, np.pi, self.n_bins + 1)
        return self

    def transform(self, X):
        lista = []
        for i in range(0, len(X[0]), self.compress):
            el = X[0, i:i + self.compress].mean()

            for minimo, massimo, lettera in zip(self.thr[:-1], self.thr[1:], iter(self.symbols)):
                if minimo < el and massimo >= el:
                    lista.append(lettera)
                    break

        return np.array(lista).reshape((1, -1))