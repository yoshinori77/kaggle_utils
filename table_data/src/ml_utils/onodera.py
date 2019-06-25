import gc
from multiprocessing import cpu_count
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


class Onodera(object):
    def __init__(self, params=None, NFOLD=10, NROUND=1600, SEED=42):
        if params is None:
            self.params = {
                'bagging_freq': 5,
                'bagging_fraction': 1.0,
                'boost_from_average': 'false',
                'boost': 'gbdt',
                'feature_fraction': 1.0,
                'learning_rate': 0.005,
                'max_depth': -1,
                'metric': 'binary_logloss',
                'min_data_in_leaf': 30,
                'min_sum_hessian_in_leaf': 10.0,
                'num_leaves': 64,
                'num_threads': cpu_count(),
                'tree_learner': 'serial',
                'objective': 'binary',
                'verbosity': -1
            }
        else:
            self.params = params
        self.NFOLD = NFOLD
        self.NROUND = NROUND
        self.SEED = SEED
        np.random.seed(SEED)
        self.var_len = None

    def __save(self, obj, file_path):
        with open(file_path, mode='wb') as f:
            pickle.dump(obj, f)

    def count_round_encoding(self, X):
        self.var_len = X.shape[1]
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

    def vstack(self, X, y_train):
        X_train_concat = np.concatenate([
            np.concatenate([
                X[:200000, 5*cnum:5*cnum+5],
                np.ones((len(y_train), 1)).astype("int")*cnum
            ], axis=1) for cnum in range(self.var_len)], axis=0)
        y_train_concat = np.concatenate(
            [y_train for cnum in range(self.var_len)], axis=0)
        return X_train_concat, y_train_concat

    def unique_y_df(self, X_train_concat, y_train_concat):
        train_group = np.arange(len(X_train_concat)) % 200000
        id_y = pd.DataFrame(zip(train_group, y_train_concat),
                            columns=['id', 'y'])
        id_y_uq = id_y.drop_duplicates('id').reset_index(drop=True)
        return id_y, id_y_uq

    def stratified(self, id_y, id_y_uq):
        id_y_uq0 = id_y_uq[id_y_uq.y == 0].sample(frac=1)
        id_y_uq1 = id_y_uq[id_y_uq.y == 1].sample(frac=1)

        id_y_uq0['g'] = [i % self.NFOLD for i in range(len(id_y_uq0))]
        id_y_uq1['g'] = [i % self.NFOLD for i in range(len(id_y_uq1))]
        id_y_uq_ = pd.concat([id_y_uq0, id_y_uq1])

        id_y_ = pd.merge(id_y[['id']], id_y_uq_, how='left', on='id')

        train_idx_list = []
        valid_idx_list = []
        for i in range(self.NFOLD):
            train_idx = id_y_[id_y_.g != i].index
            train_idx_list.append(train_idx)
            valid_idx = id_y_[id_y_.g == i].index
            valid_idx_list.append(valid_idx)

        return train_idx_list, valid_idx_list

    def kfold_train(self, id_y, train_idx_list, valid_idx_list,
                    X_train_concat, y_train_concat, X, y_train):
        NROUND = 1600
        models = []
        oof = np.zeros(len(id_y))
        p_test_all = np.zeros((100000, self.var_len, self.NFOLD))
        id_y['var'] = np.concatenate(
            [np.ones(200000)*i for i in range(self.var_len)])

        for i in range(self.NFOLD):

            print(f'building {i}...')

            train_idx = train_idx_list[i]
            valid_idx = valid_idx_list[i]

            # train
            X_train_cv = X_train_concat[train_idx]
            y_train_cv = y_train_concat[train_idx]

            # valid
            X_valid = X_train_concat[valid_idx]

            # test
            X_test = np.concatenate([
                np.concatenate([
                    X[200000:, 5*cnum:5*cnum+5],
                    np.ones((100000, 1)).astype("int")*cnum
                ], axis=1) for cnum in range(self.var_len)], axis=0
            )

            dtrain = lgb.Dataset(
                X_train_cv, y_train_cv,
                feature_name=['value', 'count_org', 'count_2',
                              'count_3', 'count_4', 'varnum'],
                categorical_feature=['varnum'], free_raw_data=False
            )
            model = lgb.train(self.params, train_set=dtrain,
                              num_boost_round=NROUND, verbose_eval=100)
            l = valid_idx.shape[0]

            p_valid = model.predict(X_valid)
            p_test = model.predict(X_test)
            for j in range(self.var_len):
                oof[valid_idx] = p_valid
                p_test_all[:, j, i] = p_test[j*100000:(j+1)*100000]

            models.append(model)
        self.__save(models, './models.p')

        id_y['pred'] = oof
        oof = pd.pivot_table(id_y, index='id', columns='var',
                             values='pred').values

        auc = f'seed{self.SEED} AUC(all var):' \
              f'{roc_auc_score(y_train, (9 * oof / (1 - oof)).prod(axis=1))}'
        print(auc)

        l = y_train.shape[0]
        oof_odds = np.ones(l) * 1 / 9
        for j in range(self.var_len):
            if roc_auc_score(y_train, oof[:, j]) >= 0.500:
                oof_odds *= (9 * oof[:, j] / (1 - oof[:, j]))

        auc = f'seed{self.SEED} AUC(th0.5): {roc_auc_score(y_train, oof_odds)}'
        print(auc)
        self.__save(oof_odds, 'oof_odds.p')
        return oof_odds

    def predict(self, p_test_all, y_train, oof, test):
        p_test_mean = p_test_all.mean(axis=2)

        p_test_odds = np.ones(100000) * 1 / 9
        for j in range(var_len):
            if roc_auc_score(y_train, oof[:, j]) >= 0.500:
                p_test_odds *= (
                    9 * p_test_mean[:, j] / (1 - p_test_mean[:, j]))

        p_test_odds = p_test_odds / (1 + p_test_odds)

        sub1 = pd.read_csv("../input/sample_submission.csv.zip")
        sub2 = pd.DataFrame(
            {"ID_code": test.ID_code.values, "target": p_test_odds})
        sub = pd.merge(sub1[["ID_code"]], sub2, how="left").fillna(0)
        return sub
