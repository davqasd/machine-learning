#! /usr/bin/env python
# -*- coding: utf-8 -*-

# sudo apt install graphviz
# pip install sklearn graphviz

from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)
import graphviz
print iris.target
dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("iris")
