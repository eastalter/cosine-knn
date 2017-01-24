#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import heapq

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


class CosineKNN():
    def __init__(self, n_neigbors=5):
        self.n_neigbors = n_neigbors

    def fit(self, X, y):
        self.data = np.array(X)
        self.train = np.array(y)

    def predict(self, X):
        self.simtree = []
        dist_dic = {}
        dic = {}
        near_labels = []
        X = np.array([X])
        X.reshape(-1, 1)
        # make heap
        for data in self.data:
            sim = pairwise_distances(X, np.array(data).reshape(1, -1),
                    metric='cosine')
            dist_dic[float(sim)] = data
            heapq.heappush(self.simtree, sim)

        # make near labels list
        for i in xrange(self.n_neigbors):
            min_sim = heapq.heappop(self.simtree)
            vector = dist_dic[float(min_sim)]
            for index, item in enumerate(self.data):
                if (item == vector).all():
                    break
            near_labels.append(self.train[index])

        # search max label
        for label in near_labels:
            dic[label] = 0
        for label in near_labels:
            dic[label] += 1
        y_pred = self.value_to_key(dic, max(dic.values()))
        return y_pred

    def value_to_key(self, dic, value):
        for key, val in zip(dic.keys(), dic.values()):
            if val == value:
                return key

if __name__ == '__main__':
    I = CosineKNN(n_neigbors=2)
    X = [[0.5, 0.86], [0.86, 0.5], [-0.5, -0.86], [-0.86, -0.5]]
    y = [1, 1, 0, 0]
    I.fit(X, y)
    print 'answer:', I.predict([0.7, 0.7])
    print 'answer:', I.predict([-0.7, -0.7])
