#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from random import sample
import time

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

# ignore Runtime Warning about divide
np.seterr(divide='ignore', invalid='ignore')

class RandomForest:
    def __init__(self, n_classifiers=30):
        self.n_classifiers = n_classifiers
        self.classifiers = []
        self.x = None
        self.y = None

    def build_tree(self, tree):
        n = len(self.y)  # n for bootstrap sampling size
        n = int(n * 0.3) + 1
        x, y = resample(self.x, self.y, n_samples=n)  # bootstrap sampling
        tree.fit(x, y)
        return tree  # return tree for multiprocessing pool

    def fit(self, x, y):
        self.x, self.y = x, y
        n_select_features = int(np.sqrt(x.shape[1]))  # number of features
        for i in range(self.n_classifiers):
            tree = DecisionTreeClassifier(criterion='gini', max_features=n_select_features) # !!!
            self.classifiers.append(self.build_tree(tree))

    def predict(self, x_test): # ensemble
        pred = []
        for tree in self.classifiers:
            y_pred = tree.predict(x_test)
            pred.append(y_pred)
        pred = np.array(pred)
        result = [Counter(pred[:, i]).most_common()[0][0] for i in range(pred.shape[1])]
        return result


def get_data(filename, target, features, sep):
    train = pd.read_csv(filename, sep=sep)
    train = train.apply (pd.to_numeric, errors='coerce')
    train = train.dropna()

    # train = train[:50] # cut x
    Y = train.pop(target).values
    x = train[features]
    return [x, Y]

def forest_kfold_cross_val(points, classes, min, max, step, metric):
    curr = min
    C = []
    res = []
    while (curr <= max):
        print "Top: %s Cur: %s" % ((max + 0.0000001), curr)
        pipe = make_pipeline(MinMaxScaler(feature_range=[0, 1]), RandomForest(n_classifiers=curr))
        scores = cross_val_score(pipe, points, classes, cv = 10, scoring = metric)
        print scores
        avg = sum(scores) / len(scores)
        C.append(curr)
        res.append(avg)
        curr += step
    draw_firest_res(C, res, metric)

def draw_firest_res(C, res, metric):
    plt.figure(figsize = (10, 10))
    plt.xlim(min(C), max(C))
    plt.ylim(min(res), max(res))
    plt.xlabel('n_classifiers')
    plt.ylabel('value')
    plt.title(metric)
    x0 = 0
    y0 = 0
    x2 = 0
    y2 = 0
    for i in range(len(C)):
        x1, y1 = [x0, C[i]], [y0, res[i]]
        plt.plot(x1, y1, marker = 'o', color = 'Green')
        x0 = C[i]
        y0 = res[i]
    plt.show()

def main():
    filename = 'wdbc.data'
    target = 'Outcome'
    features = ['id','Time',"n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9", "n10", "n11",
                "n12", "n13", "n14", "n15", "n16", "n17", "n18", "n19", "n20", "n21",
                "n22", "n23", "n24", "n25", "n26", "n27", "n28", "n29"]
    sep = ','

    x, Y = get_data(filename, target, features, sep)

    forest_kfold_cross_val(x, Y, 1, 30, 2, 'accuracy')
    # forest_kfold_cross_val(x, Y, 1, 100, 8, 'precision')
    # forest_kfold_cross_val(x, Y, 1, 100, 8, 'recall')

if __name__ == "__main__": main()
