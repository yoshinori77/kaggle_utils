from sklearn.preprocessing import LabelEncoder


class CategoryEncoding(object):
    def __init__(self, data, method='label',
                 is_target=False, target_col='target', cols=None):
        self.data = data
        self.method = method
        self.is_target = is_target
        self.target_col = target_col
        if cols is None:
            self.cols = data.select_dtypes(
                include=['object', 'category']).columns
        else:
            self.cols = cols

    def execute(self):
        if self.method == 'label':
            return self._label_encoding()
        elif self.method == 'count':
            return self._count_encoding()
        elif self.method == 'rank':
            return self._rank_encoding()
        else:
            return self._one_hot_encoding()

    def _label_encoding(self):
        '''カテゴリカル変数を数的変数に変換
        '''
        for c in self.cols:
            lbl = LabelEncoder()
            self.data[c] = lbl.fit_transform(list(self.data[c].values))
        return self.data

    def _count_encoding(self):
        def _count(col, data, is_target, target_col):
            if is_target:
                pass
            else:
                target_col = col

            return data.groupby(
                col)[target_col].transform('count')

        for c in self.cols:
            self.data[c] = _count(c, self.data,
                                  self.is_target, self.target_col)
        return self.data

    def _rank_encoding(self):
        def _rank(col, data, is_target, target_col):
            if is_target:
                pass
            else:
                target_col = col

            count_rank = data.groupby(col)[
                target_col].count().rank(ascending=False)
            return data[col].map(count_rank)

        for c in self.cols:
            self.data[c] = _rank(c, self.data,
                                 self.is_target, self.target_col)
        return self.data

    def _one_hot_encoding(self):
        return self.data.get_dummies(columns=self.cols, drop_first=True)
