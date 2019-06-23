import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from fancyimpute import SimpleFill, KNN, SoftImpute, IterativeSVD


class IterativeInterpolate(BaseEstimator, TransformerMixin):
    def __init__(self, estimater=None, is_estimate=False,
                 missing_values=np.nan, max_iter=10, random_state=None):
        self.estimater = estimater
        self.is_estimate = is_estimate
        self.missing_values = missing_values
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y=None):
        if self.is_estimate:
            self.imp = IterativeImputer(
                estimator=self.estimater, missing_values=self.missing_values,
                max_iter=self.max_iter, random_state=self.random_state)
        else:
            self.imp = IterativeImputer(
                missing_values=self.missing_values,
                max_iter=self.max_iter, random_state=self.random_state)
        if y is None:
            self.imp.fit(X)
        else:
            self.imp.fit(X, y)
        return self

    def transform(self, X):
        return self.imp.transform(X)
