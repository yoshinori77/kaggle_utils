import numpy as np
import pandas as pd
import scipy.sparse


class IncidenceMatrix(object):
    def __init__(self, data, first_id, second_id, **kwargs):
        self.data = data
        self.first_id = first_id
        self.second_id = second_id

    def execute(self):
        self._make_sparse_matrix()
        return self._count_pair()

    def _make_sparse_matrix(self):
        i = np.unique(np.concatenate(
                [self.data[[self.first_id, self.second_id]].values,
                 self.data[[self.second_id, self.first_id]].values]), axis=0)
        sparse_matrix = scipy.sparse.coo_matrix(
            (np.ones(len(i)), (i[:, 0], i[:, 1])))
        sparse_matrix = sparse_matrix.tocsr()
        self.sparse_matrix = sparse_matrix

    def _count_pair(self):
        rows_FirstId = self.sparse_matrix[self.data[self.first_id].values]
        rows_SecondId = self.sparse_matrix[self.data[self.second_id].values]
        f = np.array(rows_FirstId.multiply(
            rows_SecondId).sum(axis=1)).squeeze()
        return np.unique(f, return_counts=True)
