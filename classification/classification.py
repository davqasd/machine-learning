#! /usr/bin/env python
# -*- coding: utf-8 -*-

# pip install -U scikit-learn

import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib import colors as mcolors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

# ==========================================================
def test_methods(x, Y):
    # ======================
    model = GaussianNB()
    model.fit(x, Y)
    # predicted = model.predict([[50, 21, 77, 0, 28, 0, 27, 48, 22]])
    # print predicted
    # ======================
    model = LogisticRegression(random_state=0)
    model.fit(x, Y)
    # predicted = model.predict([[50, 21, 77, 0, 28, 0, 27, 48, 22]])
    # print predicted
    # ======================
    model = SVC(gamma='auto')
    model.fit(x, Y)
    # predicted = model.predict([[50, 21, 77, 0, 28, 0, 27, 48, 22]])
    # print predicted

def show_data(filename, target, features, sep):
    train = pd.read_csv(filename, sep=sep)
    train = train[:3000] # cut x
    print train.head()
    print train.shape
    Y = train.pop(target).values
    print Y
    print Y.shape
    x = train[features]
    print x
    print x.shape
# ==========================================================

def get_data(filename, target, features, sep):
    train = pd.read_csv(filename, sep=sep)
    train = train[:3000] # cut x
    Y = train.pop(target).values
    x = train[features]
    return [x, Y]

def draw_svm_res(C, gammas, res, metric):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(min(C), max(C))
    ax.set_ylim(min(gammas), max(gammas))
    ax.set_zlim(min(res), max(res))
    ax.set_xlabel('C')
    ax.set_ylabel('gamma')
    ax.set_zlabel('res')
    ax.plot(C, gammas, res, label=metric, color = 'Red')
    plt.show()

def svm_kfold_cross_val(points, classes, C_l, C_r, C_step, G_l, G_r, G_step, metric):
    curr_C_l = C_l
    C = []
    gammas = []
    res = []
    while (curr_C_l < C_r + 0.0000001):
        curr_G_l = G_l
        print "Top C_r: %s Cur: %s" % ((C_r + 0.0000001), curr_C_l)
        while (curr_G_l < G_r + 0.0000001):
            print "Top G_r: %s Cur: %s" % ((G_r + 0.0000001), curr_G_l)
            pipe = make_pipeline(StandardScaler(), SVC(C = curr_C_l, gamma = curr_G_l))
            scores = cross_val_score(pipe, points, classes, cv = 5, scoring = metric)
            avg = sum(scores) / len(scores)
            C.append(curr_C_l)
            gammas.append(curr_G_l)
            res.append(avg)
            curr_G_l += G_step
        curr_C_l += C_step
    draw_svm_res(C, gammas, res, metric)

def draw_bayes_res(S, res, metric):
    plt.figure(figsize = (10, 10))
    plt.xlim(min(S), max(S))
    plt.ylim(min(res), max(res))
    plt.xlabel('var_smoothing')
    plt.ylabel('value')
    plt.title(metric)
    for i in range(len(S) - 1):
        x1, y1 = [S[i], S[i + 1]], [res[i], res[i + 1]]
        plt.plot(x1, y1, marker = 'o', color = 'Green')
    plt.show()

def bayes_kfold_cross_val(points, classes, S_l, S_r, S_step, metric):
    curr_S_l = S_l
    S = []
    res = []
    while (curr_S_l < S_r + 0.0000001):
        print "Top: %s Cur: %s" % ((S_r + 0.0000001), curr_S_l)
        pipe = make_pipeline(StandardScaler(), GaussianNB(var_smoothing = curr_S_l))
        scores = cross_val_score(pipe, points, classes, cv = 5, scoring = metric)
        avg = sum(scores) / len(scores)
        S.append(curr_S_l)
        res.append(avg)
        curr_S_l += S_step
    draw_bayes_res(S, res, metric)

def draw_log_reg_res(C, pen, res, metric):
    plt.figure(figsize = (10, 10))
    plt.xlim(min(C), max(C))
    plt.ylim(min(res), max(res))
    plt.xlabel('C')
    plt.ylabel('value')
    plt.title(metric)
    x0 = 0
    y0 = 0
    x2 = 0
    y2 = 0
    for i in range(len(C)):
        if (pen[i] == 'l1'):
            x1, y1 = [x0, C[i]], [y0, res[i]]
            plt.plot(x1, y1, marker = 'o', color = 'Green')
            x0 = C[i]
            y0 = res[i]
        else:
            x1, y1 = [x2, C[i]], [y2, res[i]]
            plt.plot(x1, y1, marker = 'o', color = 'Blue')
            x2 = C[i]
            y2 = res[i]
    plt.show()

def log_reg_kfold_cross_val(points, classes, C_l, C_r, C_step, Penalty, metric):
    print classes
    curr_C_l = C_l
    C = []
    P = []
    res = []
    while (curr_C_l < C_r + 0.000001):
        print "Top: %s Cur: %s" % ((C_r + 0.0000001), curr_C_l)
        for pen in Penalty:
            pipe = make_pipeline(MinMaxScaler(feature_range=[0, 1]), LogisticRegression(C = curr_C_l, penalty = pen))
            scores = cross_val_score(pipe, points, classes, cv = 5, scoring = metric)
            avg = sum(scores) / len(scores)
            C.append(curr_C_l)
            P.append(pen)
            res.append(avg)
        curr_C_l += C_step
    draw_log_reg_res(C, P, res, metric)

def main():
    filename = 'adult.data'
    target = 'income'
    features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    sep = ', '

    show_data(filename, target, features, sep)

    x, Y = get_data(filename, target, features, sep)

    test_methods(x, Y)

    # log_reg_kfold_cross_val(x, Y, 0.01, 0.99, 0.01, ['l1', 'l2'], 'accuracy')
    # log_reg_kfold_cross_val(x, Y, 0.01, 0.99, 0.01, ['l1', 'l2'], 'recall')
    # log_reg_kfold_cross_val(x, Y, 0.01, 0.99, 0.01, ['l1', 'l2'], 'precision')

    # bayes_kfold_cross_val(x, Y, 0.01, 1.0, 0.01, 'accuracy')
    # bayes_kfold_cross_val(x, Y, 0.01, 1.0, 0.01, 'recall')
    # bayes_kfold_cross_val(x, Y, 0.01, 1.0, 0.01, 'precision')

    # svm_kfold_cross_val(x, Y, 0.01, 100.0, 25.0, 0.05, 3.0, 1, 'accuracy')
    # svm_kfold_cross_val(x, Y, 0.01, 100.0, 25.0, 0.05, 3.0, 1, 'recall')
    svm_kfold_cross_val(x, Y, 0.01, 100.0, 25.0, 0.05, 3.0, 1, 'precision')

if __name__ == "__main__": main()
