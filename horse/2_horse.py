#! /usr/bin/env python
# -*- coding: utf-8 -*-

# python3 kNN.py
# sudo apt install python-numpy python-numpy python-scipy python-matplotlib python-pip
# pip install numpy scipy matplotlib more-itertools

# Как была найдена картинка:
# 1. Отбор категориальных признаков и нет
# 2. Просмотр всех картинок, где по оси X и Y - некатегориальные фичи, а цветами - категориальные

# Основной вывод работы:
# Наивысшая точность kNN для выбранных данных (по оси X - назогастральный рефлюкс PH
# по оси Y - общий белок и категории - является ли элемент данной выборки патологией)
# равняется 0.896551724138 при k = 3. k выбирался путём сравнение точностей алгоритма на
# каждом шаге k, начиная со второго при заданной дельте.
# Учитывая данную точность, можно с большой долей вероятности определять для лошади
# является ли это патологией или нет по двум параметрам.

# Негласное правило порядка параметров в функциях: 1- категория, 2 - X, 3 - Y

import csv
from itertools import combinations # для combinations
from math import sqrt
# Либы для отрисовки
import pylab as pl
import numpy as np
from matplotlib.colors import ListedColormap

# Нельзя просто убрать все строки с ? - иначе не останется данных, нужно
# очищать данные только в выбранных столбцах одновременно
class Parser:
    def __init__(self, filename):
        f = open(filename, "r")
        self.parsed_file = list(csv.reader(f.readlines(), delimiter = ' '))
    def get_column(self, n):
        col = []
        for row in self.parsed_file:
            col.append(row[n - 1])
        return col
    # создание фильтрованных данных в виде столбцов и колонок
    # в 2 раза больше памяти, зато не надо тратить время на транспонирование матрицы
    def create_data_with_filter(self, array_indexes):
        self.data_rows = []
        self.data_columns = []
        for i in list(range(0, len(array_indexes))):
            self.data_columns.append([])
        for row in self.parsed_file:
            def doWork(): # функция в функции - чтобы выйти из двумерного цикла
                vals = []
                for index in array_indexes:
                    val = row[index - 1]
                    if val == '?': # очистка данных
                        return 'None'
                    vals.append(float(val))
                return vals
            vals = doWork()
            if vals == 'None':
                continue
            self.data_rows.append(vals)
            i = 0
            for val in vals:
                self.data_columns[i].append(val)
                i += 1

class MyMachineLearning:
    # Передаём номера столбцов
    def __init__(self, category_n, n1, n2):
        self.train_data_parser = Parser("horse-colic.data")
        self.train_data_parser.create_data_with_filter([category_n, n1, n2])
        self.train_data_rows = self.train_data_parser.data_rows
        self.test_data_parser = Parser("horse-colic.test")
        self.test_data_parser.create_data_with_filter([n1, n2])
        self.test_data_rows = self.test_data_parser.data_rows
         # для проверки точности
        self.test_data_parser.create_data_with_filter([category_n, n1, n2])
        self.test_data_with_category_rows = self.test_data_parser.data_rows
    def show_train_data(self):
        colors = ['#d37b4c', '#9dd159', '#c49a31', '#dcbb99', '#925378']
        colors += ['#0ab33b', '#5087e8']
        classColormap = ListedColormap(colors)
        train_data = self.train_data_parser.data_columns
        pl.scatter(train_data[1],
                   train_data[2],
                   c=train_data[0],
                   cmap=classColormap)
        pl.title('28: is pathology data present for this case? (color)')
        pl.xlabel('16: nasogastric reflux PH')
        pl.ylabel('20: total protein')
        pl.show()
    def count_uniq_first_column(self, a):
        buf = []
        for row in a:
            buf.append(row[0])
        return len(list(set(buf)))
    def classifyKNN (self, trainData, testData, k):
        numberOfClasses = self.count_uniq_first_column(trainData)
        def dist (a, b):
            return sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
        testLabels = []
        for testPoint in testData:
            # Вычисление расстояния между тестовой точкой и всеми тренировачными точками
            testDist = [ [dist(testPoint, [trainData[i][1], trainData[i][2]]), trainData[i][0]] for i in range(len(trainData))]
            # Как много точек каждого класса среди ближайших K точек
            stat = [0 for i in range(numberOfClasses)]
            for d in sorted(testDist)[0:k]:
                stat[int(d[1]) - 1] += 1
            # Принятие решения о том, к какому классу относится тестовая точка
            testLabels.append( sorted(zip(stat, range(numberOfClasses)), reverse=True)[0][1] + 1 )
        return testLabels
    def calculateAccuracy (self, k):
        trainData = self.train_data_rows
        testDataWithLabels = self.test_data_with_category_rows
        testData = self.test_data_rows
        testDataLabels = self.classifyKNN (trainData, testData, k)
        accuracy = sum([int(testDataLabels[i]==testDataWithLabels[i][0]) for i in range(len(testDataWithLabels))]) / float(len(testDataWithLabels))
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
        if current_delta <= 0:
            k -= 1
        return k
    # Отрисовка работы kNN
    def showDataOnMesh (self, k):
        def generateTestMesh (trainData):
            # trainData[i]: 0 - category, 1 - X, 2 - Y
            x_min = min( [trainData[i][1] for i in range(len(trainData))] ) - 1.0
            x_max = max( [trainData[i][1] for i in range(len(trainData))] ) + 1.0
            y_min = min( [trainData[i][2] for i in range(len(trainData))] ) - 1.0
            y_max = max( [trainData[i][2] for i in range(len(trainData))] ) + 1.0
            h = 0.25
            testX, testY = np.meshgrid(np.arange(x_min, x_max, h),
                                       np.arange(y_min, y_max, h))
            return [testX, testY]
        trainData      = self.train_data_rows
        testMesh       = generateTestMesh (trainData)
        testMeshLabels = self.classifyKNN (trainData, zip(testMesh[0].ravel(), testMesh[1].ravel()), k)
        classColormap  = ListedColormap(['#d37b4c', '#00FF00', '#FF00FF', '#FF0000'])
        testColormap   = ListedColormap(['#925378', '#5087e8', '#FFFF00', '#0000FF'])
        pl.pcolormesh(testMesh[0],
                      testMesh[1],
                      np.asarray(testMeshLabels).reshape(testMesh[0].shape),
                      cmap=testColormap)
        pl.scatter([trainData[i][1] for i in range(len(trainData))],
                   [trainData[i][2] for i in range(len(trainData))],
                   c=[trainData[i][0] for i in range(len(trainData))],
                   cmap=classColormap)
        pl.title('28: is pathology data present for this case? (color)\nshow kNN territory')
        pl.xlabel('16: nasogastric reflux PH')
        pl.ylabel('20: total protein')
        pl.show()

# Проверка выбранных столбцов
parser = Parser("horse-colic.data")
parser.create_data_with_filter([1, 16, 20])
print parser.data_columns
print parser.data_rows

categories = [1, 2, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 21, 23, 24, 25, 26, 27, 28]
non_cat = [3, 4, 5, 6, 16, 19, 20, 22]

# 1. Категория | 2. X | 3. Y
machine = MyMachineLearning(28, 20, 22)
k = machine.loo_find_k(0.000000001)
accuracy = machine.calculateAccuracy(k)
print "k по LOO:", k
print "accuracy:", accuracy

machine.show_train_data()
print machine.classifyKNN(machine.train_data_rows, machine.test_data_rows, k)
machine.showDataOnMesh(k)
