from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor


class FeatureSelect(object):
    def __init__(self, X, y, is_label=False, n=400, max_depth=5, alpha=0.05):
        self.X = X
        self.y = y
        self.is_label = is_label
        self.n = n
        self.max_depth = max_depth
        self.alpha = alpha

    def execute(self):
        if self.is_label:
            return self._feature_select_label_boruta()
        else:
            return self._feature_select_boruta()

    def _feature_select_boruta(self):
        '''borutaで特徴選択した特徴量を返す
        '''
        rf = RandomForestRegressor(
            n_estimators=self.n, max_depth=self.max_depth,
            random_state=42, n_jobs=-1)
        feat_selector = BorutaPy(rf, n_estimators='auto',
                                 alpha=self.alpha, verbose=2, random_state=1)
        feat_selector.fit(self.X.values, self.y.values)
        return feat_selector.transform(self.X.values)

    def _feature_select_label_boruta(self):
        '''borutaで特徴選択した特徴量のラベルを返す
        '''
        rf = RandomForestRegressor(
            n_estimators=self.n, max_depth=self.max_depth,
            random_state=42, n_jobs=-1)
        feat_selector = BorutaPy(rf, n_estimators='auto',
                                 alpha=self.alpha, verbose=2, random_state=1)
        feat_selector.fit(self.X.values, self.y.values)
        return self.X.columns[feat_selector.support_]
