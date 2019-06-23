from copy import deepcopy

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest


class LofDetection(BaseEstimator, TransformerMixin):
    def __init__(self, contamination=0):
        self.contamination = contamination

    def fit(self, X, y=None):
        if self.contamination == 0:
            return self
        self.lof = LocalOutlierFactor(
            contamination=self.contamination, novelty=True)
        if y is None:
            self.lof.fit(X)
        else:
            self.lof.fit(X, y)

        return self

    def transform(self, X_):
        X = deepcopy(X_)
        if self.contamination == 0:
            return X
        idx_outlier = self.lof.predict(X) == -1
        X[idx_outlier, :] = np.nan

        simple_imputer = SimpleImputer()
        X = simple_imputer.fit_transform(X)

        return X


class EllipticDetection(BaseEstimator, TransformerMixin):
    def __init__(self, contamination=0):
        self.contamination = contamination

    def fit(self, X, y=None):
        if self.contamination == 0:
            return self
        self.ell = EllipticEnvelope(
            contamination=self.contamination)
        if y is None:
            self.ell.fit(X)
        else:
            self.ell.fit(X, y)

        return self

    def transform(self, X_):
        X = deepcopy(X_)
        if self.contamination == 0:
            return X
        idx_outlier = self.ell.predict(X) == -1
        X[idx_outlier, :] = np.nan

        simple_imputer = SimpleImputer()
        X = simple_imputer.fit_transform(X)

        return X


class IsolationForestDetection(BaseEstimator, TransformerMixin):
    def __init__(self, contamination=0):
        self.contamination = contamination

    def fit(self, X, y=None):
        if self.contamination == 0:
            return self
        self.iso = IsolationForest(
            contamination=self.contamination)
        if y is None:
            self.iso.fit(X)
        else:
            self.iso.fit(X, y)

        return self

    def transform(self, X_):
        X = deepcopy(X_)
        if self.contamination == 0:
            return X
        idx_outlier = self.iso.predict(X) == -1
        X[idx_outlier, :] = np.nan

        simple_imputer = SimpleImputer()
        X = simple_imputer.fit_transform(X)

        return X
