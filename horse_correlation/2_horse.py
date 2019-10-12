#! /usr/bin/env python
# -*- coding: utf-8 -*-

# python3 kNN.py
# sudo apt install python-numpy python-numpy python-scipy python-matplotlib python-pip
# pip install numpy scipy matplotlib more-itertools

# 1. Транспонировать матрицу
# 2. Визуализировать kNN
# 3. Убрать корреляцию
# 4. Сделать логичные выводы на экран
# 5. Сделать выводы о выборке простецкие
# 6. Расписать что к чему

# Как была найдена картинка:
# 1. Отбор категориальных признаков и нет
# 2. Просмотр всех картинок, где по оси X и Y - некатегориальные фичи, а цветами - категориальные

import csv
from itertools import combinations # для combinations
from math import sqrt
# Либы для отрисовки
import pylab as pl
import numpy as np
from matplotlib.colors import ListedColormap

# Нахождение корреляции двух массивов
def r(a1, a2):
    mean_x = sum(a1) / len(a1)
    mean_y = sum(a2) / len(a2)
    cov = 0
    for a in zip(a1, a2):
        cov += (a[0] - mean_x) * (a[1] - mean_y)
    var_x = 0
    for a in a1:
        var_x += (a - mean_x) ** 2
    var_y = 0
    for a in a2:
        var_y += (a - mean_y) ** 2
    if (var_x == 0 or var_y == 0):
        return 'NaN'
    r = cov / sqrt(var_x)
    r /= sqrt(var_y)
    return r

# Нельзя просто убрать все строки с ? - иначе не останется данных, нужно
# очищать данные только в выбранных столбцах одновременно
class Parser:
    def __init__(self, filename):
        f = open(filename, "r")
        self.parsed_file = list(csv.reader(f.readlines(), delimiter = ' '))
        self.result = []
    # Находим корреляции каждого столбца
    def correlations(self):
        n_cols = len(self.parsed_file[0])
        # 378 проходов - для 28 столбцов (около 0.2с)
        for n_cols in list(combinations(list(range(0, n_cols)), 2)):
            col1, col2 = [], []
            n_col1, n_col2 = n_cols[0], n_cols[1]
            for row in self.parsed_file:
                val1, val2 = row[n_col1], row[n_col2]
                if val1 == '?' or val2 == '?': # очистка данных
                    continue
                col1.append(float(val1))
                col2.append(float(val2))
            cor = r(col1, col2)
            if cor != 'NaN':
                self.result.append([cor, n_col1 + 1, n_col2 + 1])
        return sorted(self.result, key = lambda a: abs(a[0]))

    def get_column(self, n):
        col = []
        for row in self.parsed_file:
            col.append(row[n - 1])
        return col
    def get_columns_with_filter(self, array_indexes):
        cols = []
        for i in list(range(0, len(array_indexes))):
            cols.append([])
        for row in self.parsed_file:
            def doWork(): # функция в функции - чтобы выйти из двумерного цикла
                vals = []
                for index in array_indexes:
                    val = row[index - 1]
                    if val == '?': # очистка данных
                        return 'None'
                    vals.append(val)
                return vals
            vals = doWork()
            if vals == 'None':
                continue
            i = 0
            for val in vals:
                cols[i].append(float(val))
                i += 1
        return cols

class MyMachineLearning:
    # Передаём номера столбцов
    def __init__(self, category_n, n1, n2):
        self.train_data_parser = Parser("horse-colic.data")
        self.train_data = self.train_data_parser.get_columns_with_filter([category_n, n1, n2])
        self.test_data_parser = Parser("horse-colic.test")
        self.test_data = self.test_data_parser.get_columns_with_filter([n1, n2])
         # для проверки точности
        self.test_data_with_category = self.test_data_parser.get_columns_with_filter([category_n, n1, n2])
    def show_train_data(self):
        colors = ['#d37b4c', '#9dd159', '#c49a31', '#dcbb99', '#925378']
        colors += ['#0ab33b', '#5087e8']
        classColormap = ListedColormap(colors)
        pl.scatter(self.train_data[1],
                   self.train_data[2],
                   c=self.train_data[0],
                   cmap=classColormap)
        pl.show()
    def classifyKNN (self, trainData, testData, k):
        numberOfClasses = len(list(set(trainData[0])))
        def dist (a, b):
            return sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
        testLabels = []
        test_data_i = 0
        for test_x in testData[0]:
            testPoint = [test_x, testData[1][test_data_i]]
            # Вычисление расстояния между тестовой точкой и всеми тренировачными точками
            testDist = [ [dist(testPoint, [trainData[1][i], trainData[2][i]]), trainData[0][i]] for i in range(len(trainData[0]))]
            # Как много точек каждого класса среди ближайших K точек
            stat = [0 for i in range(numberOfClasses)]
            for d in sorted(testDist)[0:k]:
                stat[int(d[1]) - 1] += 1
            # Принятие решения о том, к какому классу относится тестовая точка
            testLabels.append( sorted(zip(stat, range(numberOfClasses)), reverse=True)[0][1] + 1 )
            test_data_i += 1
        return testLabels
    def calculateAccuracy (self, k):
        trainData = self.train_data
        print self.test_data_with_category[0]
        testDataWithLabels = self.test_data_with_category
        testData = self.test_data
        testDataLabels = self.classifyKNN (trainData, testData, k)
        accuracy = sum([int(testDataLabels[i]==testDataWithLabels[0][i]) for i in range(len(testDataWithLabels[0]))]) / float(len(testDataWithLabels[0]))
        return accuracy
    # параметр k определяем по критерию скользящего контроля с исключением объектов по одному (leave-one-out, LOO)
    def loo_find_k(self, delta):
        # для k = 1 - метод NN
        k = 2
        accuracy1 = self.calculateAccuracy(k)
        k += 1
        accuracy2 = self.calculateAccuracy(k)
        current_delta = accuracy2 - accuracy1
        while current_delta > delta or current_delta > 0:
            k += 1
            accuracy1 = accuracy2
            accuracy2 = self.calculateAccuracy(k)
            current_delta = accuracy2 - accuracy1
            print accuracy1, accuracy2, delta
        if current_delta <= 0:
            k -= 1
        return k
    def showDataOnMesh (nClasses, nItemsInClass, k):
        #Generate a mesh of nodes that covers all train cases
        def generateTestMesh (trainData):
            x_min = min( [trainData[i][0][0] for i in range(len(trainData))] ) - 1.0
            x_max = max( [trainData[i][0][0] for i in range(len(trainData))] ) + 1.0
            y_min = min( [trainData[i][0][1] for i in range(len(trainData))] ) - 1.0
            y_max = max( [trainData[i][0][1] for i in range(len(trainData))] ) + 1.0
            h = 0.05
            testX, testY = np.meshgrid(np.arange(x_min, x_max, h),
                                       np.arange(y_min, y_max, h))
            return [testX, testY]
        trainData      = generateData (nItemsInClass, nClasses)
        testMesh       = generateTestMesh (trainData)
        testMeshLabels = classifyKNN (trainData, zip(testMesh[0].ravel(), testMesh[1].ravel()), k, nClasses)
        classColormap  = ListedColormap(['#FF0000', '#00FF00', '#FFFFFF'])
        testColormap   = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAAA'])
        pl.pcolormesh(testMesh[0],
                      testMesh[1],
                      np.asarray(testMeshLabels).reshape(testMesh[0].shape),
                      cmap=testColormap)
        pl.scatter([trainData[i][0][0] for i in range(len(trainData))],
                   [trainData[i][0][1] for i in range(len(trainData))],
                   c=[trainData[i][1] for i in range(len(trainData))],
                   cmap=classColormap)
        pl.show()

parser = Parser("horse-colic.data")
for row in parser.correlations():
    print row

print parser.get_column(1)

# Проверка выбранных столбцов
max_r_columns = parser.get_columns_with_filter([16, 20])
print r(max_r_columns[0], max_r_columns[1])

print parser.get_columns_with_filter([1, 7, 8])

categories = [1, 2, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 21, 23, 24, 25, 26, 27, 28]
non_cat = [3 - 6, 16, 19, 20, 22]

# 1. Категория | 2. X | 3. Y
machine = MyMachineLearning(28, 20, 22)
# machine.show_data()
print machine.classifyKNN(machine.train_data, machine.test_data, 10)
machine.show_train_data()
k = machine.loo_find_k(0.000000001)
accuracy = machine.calculateAccuracy(3)
print "k по LOO:", k
print "accuracy:", accuracy
