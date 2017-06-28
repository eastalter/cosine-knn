#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


class CosineKNN():
    def __init__(self, n_neigbors=5):
        self.n_neigbors = n_neigbors

    def fit(self, X, y):
        self.data = np.array(X)
        self.train = np.array(y)

    def set_sim(self, prediction):
        self.sim = pairwise_distances(self.data, np.array(prediction), metric='cosine')
        print self.sim.T

    def knn(self):
        # 類似度行列から、類似度の高い順に多いラベルをピックアップする
        result = []
        for aff in self.sim.T:
            _, index = self.descending_order(aff)
            label_counter = [0 for __ in range(max(self.train) + 1)]
            for item in index:
                label_counter[self.train[item]] += 1
            label = label_counter.index(max(label_counter))
            result.append(label)
        return result

    def descending_order(self, array):
        index = []
        data = []
        for i in range(len(array)):
            data.append(np.sort(array)[::][i])
            index.append(np.argsort(array)[::][i])
        return data[:self.n_neigbors], index[:self.n_neigbors]

    def value_to_key(self, dic, value):
        for key, val in zip(dic.keys(), dic.values()):
            if val == value:
                return key

if __name__ == '__main__':
    I = CosineKNN(n_neigbors=1)
    X = [[0.5, 0.86], [0.86, 0.5], [-0.5, -0.86], [-0.86, -0.5]]
    y = [0, 3, 2, 1]
    I.fit(X, y)
    I.set_sim([[0.1, 0.2], [-0.5, -0.8], [0.8, 0.5]])
    print I.knn()
