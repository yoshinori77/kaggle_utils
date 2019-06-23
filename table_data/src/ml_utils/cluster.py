from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


class Kmeans(BaseEstimator, ClusterMixin, TransformerMixin):
    def __init__(self, n_clusters=8, max_iter=300,
                 verbose=0, random_state=None, n_jobs=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.verbose = verbose
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        self.kmeans = KMeans(n_clusters=self.n_clusters,
                             max_iter=self.max_iter,
                             verbose=self.verbose,
                             random_state=self.random_state,
                             n_jobs=self.n_jobs)
        if y is None:
            self.kmeans.fit(X)
        else:
            self.kmeans.fit(X, y)

        return self

    def predict(self, X):
        return self.kmeans.predict(X)

    def transform(self, X):
        return self.kmeans.transform(X)


class GMM(BaseEstimator, ClusterMixin):
    def __init__(self, n_components=1, max_iter=100, n_init=1,
                 random_state=None, verbose=0, verbose_interval=10):
        self.n_components = n_components
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        self.verbose = verbose
        self.verbose_interval = verbose_interval

    def fit(self, X, y=None):
        self.gmm = GaussianMixture(
            n_components=self.n_components, max_iter=self.max_iter,
            n_init=self.n_init, random_state=self.random_state,
            verbose=self.verbose, verbose_interval=self.verbose_interval)
        if y is None:
            self.gmm.fit(X)
        else:
            self.gmm.fit(X, y)

        return self

    def predict(self, X):
        return self.gmm.predict(X)

    def predict_proba(self, X):
        return self.gmm.predict_proba(X)
