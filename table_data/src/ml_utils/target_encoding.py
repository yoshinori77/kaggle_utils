from itertools import product

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


class TargetEncoding(object):
    def __init__(self, data, group, target='target', method='kfold'):
        self.data = data
        self.group = group
        self.target = target
        self.method = method

    def execute(self):
        if self.method == 'kfold':
            return self._kfold()
        elif self.method == 'leave_one_out':
            return self._leave_one_out()
        elif self.method == 'smoothing':
            return self._smoothing()
        else:
            return self._expanding()

    def _output(self):
        encoded_feature = self.data['target_enc'].values
        corr = np.corrcoef(
            self.data[self.target].values, encoded_feature)[0][1]
        return encoded_feature, corr

    def _kfold(self):
        kf = KFold(n_splits=5, shuffle=False)

        for tr_ind, val_ind in kf.split(self.data):
            X_tr, X_val = self.data.iloc[tr_ind], self.data.iloc[val_ind]
            X_val['target_enc'] = X_val[self.group].map(
                X_tr.groupby(self.group)[self.target].mean())
            self.data.iloc[val_ind] = X_val

        return self._output()

    def _leave_one_out(self):
        target_sum = self.data.groupby(
            self.group)[self.target].transform('sum')
        n_objects = self.data.groupby(
            self.group)[self.target].transform('count')
        self.data['target_enc'] = (
            target_sum - self.data[self.target]) / (n_objects - 1)
        return self._output()

    def _smoothing(self):
        item_id_target_mean = self.data.groupby(
            self.group)[self.target].transform('mean')
        n_objects = self.data.groupby(
            self.group)[self.target].transform('count')
        self.data['target_enc'] = (
            item_id_target_mean * n_objects) / (n_objects + 100)
        return self._output()

    def _expanding(self):
        cumsum = self.data.groupby(
            self.group)[self.target].cumsum() - self.data[self.target]
        cumcnt = self.data.groupby(self.group).cumcount()
        self.data['target_enc'] = cumsum / cumcnt
        return self._output()
