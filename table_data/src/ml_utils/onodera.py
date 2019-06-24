import numpy as np
import pandas as pd
from tqdm import tqdm
import gc


class Onodera(object):
    def __init__(self):
        pass

    def count_round_encoding(self, X):
        var_len = X.shape[1]
        X_cnt = np.zeros((len(X), var_len * 5))

        for j in tqdm(range(var_len)):
            for i in range(1, 4):
                x = np.round(X[:, j], i+1)
                dic = pd.value_counts(x).to_dict()
                X_cnt[:, i+j*4] = pd.Series(x).map(dic)
            x = X[:, j]
            dic = pd.value_counts(x).to_dict()
            X_cnt[:, j*4] = pd.Series(x).map(dic)

        X_raw = X.copy()
        del X
        gc.collect()

        X = np.zeros((len(X_raw), var_len * 5))
        for j in tqdm(range(var_len)):
            X[:, 5*j+1:5*j+5] = X_cnt[:, 4*j:4*j+4]
            X[:, 5*j] = X_raw[:, j]

        return X

    def unpivot_all_vars(self, X, y_train, var_len):
        X_train_concat = np.concatenate([
            np.concatenate([
                X[:200000, 5*cnum:5*cnum+5],
                np.ones((len(y_train), 1)).astype("int")*cnum
            ], axis=1) for cnum in range(var_len)], axis=0)
        y_train_concat = np.concatenate(
            [y_train for cnum in range(var_len)], axis=0)
        return X_train_concat, y_train_concat

    def unique_y_df(self, X_train_concat, y_train_concat):
        train_group = np.arange(len(X_train_concat)) % 200000
        id_y = pd.DataFrame(zip(train_group, y_train_concat),
                            columns=['id', 'y'])
        id_y_uq = id_y.drop_duplicates('id').reset_index(drop=True)
        return id_y, id_y_uq

    def stratified(self, id_y, id_y_uq, nfold=5):
        id_y_uq0 = id_y_uq[id_y_uq.y == 0].sample(frac=1)
        id_y_uq1 = id_y_uq[id_y_uq.y == 1].sample(frac=1)

        id_y_uq0['g'] = [i % nfold for i in range(len(id_y_uq0))]
        id_y_uq1['g'] = [i % nfold for i in range(len(id_y_uq1))]
        id_y_uq_ = pd.concat([id_y_uq0, id_y_uq1])

        id_y_ = pd.merge(id_y[['id']], id_y_uq_, how='left', on='id')

        train_idx_list = []
        valid_idx_list = []
        for i in range(nfold):
            train_idx = id_y_[id_y_.g != i].index
            train_idx_list.append(train_idx)
            valid_idx = id_y_[id_y_.g == i].index
            valid_idx_list.append(valid_idx)

        return train_idx_list, valid_idx_list
