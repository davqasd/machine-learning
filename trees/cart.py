#! /usr/bin/env python
# -*- coding: utf-8 -*-

# python cart.py
import csv
import json
import math # для log
import os # для красивого print

# sudo apt install graphviz
# pip install sklearn graphviz pandas
from sklearn import tree
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
import graphviz

# Нельзя просто убрать все строки с ? - иначе не останется данных, нужно
# очищать данные только в выбранных столбцах одновременно
class Parser:
    def __init__(self, data_filename, headers_filename, columns_to_project):
        self.data_filename = data_filename
        self.headers_filename = headers_filename
        self.columns_to_project = columns_to_project
        self.init_data()
    def init_data(self):
        f = open(self.data_filename, "r")
        parsed_file = list(csv.reader(f.readlines(), delimiter = ' '))
        headers_data = open(self.headers_filename, "r").read()
        self.headers = []
        data_values = json.loads(headers_data)["headers"]
        for header in data_values:
            self.headers.append(header.keys()[0])

        data = { 'headers': self.headers, 'rows': parsed_file }
        idx_to_name, name_to_idx = self.get_header_name_to_idx_maps(self.headers)

        project_column_ids = [name_to_idx[name] for name in self.columns_to_project]

        rows = self.create_data_with_filter(parsed_file, project_column_ids, data_values)
        idx_to_name, name_to_idx = self.get_header_name_to_idx_maps(self.columns_to_project)
        self.data = {
            'header': self.columns_to_project,
            'rows': rows,
            'name_to_idx': name_to_idx,
            'idx_to_name': idx_to_name
        }
    # создание фильтрованных данных в виде строк
    def create_data_with_filter(self, parsed_file, array_indexes, data_values):
        data_rows = []
        for row in parsed_file:
            def doWork(): # функция в функции - чтобы выйти из двумерного цикла
                vals = []
                for index in array_indexes:
                    val = row[index]
                    if val == '?' or not val: # очистка данных
                        return 'None'
                    # Важно! Преобразование числовых данных из horse-colic.data в horse-colic-headers.json
                    # Например 1-й столбец 1-е значение в horse-colic.data = 2
                    # Из horse-colic.names узнаём, что 1 = Yes, it had surgery, 2 = It was treated without surgery
                    # Следовательно в vals попадает "It was treated without surgery".
                    # vals.append(data_values[index].values()[0][int(val) - 1]) # значения с 1, индексация массива с 0
                    vals.append(int(val) - 1)
                return vals
            vals = doWork()
            if vals == 'None':
                continue
            data_rows.append(vals)
        return data_rows
    # Метод маппит названия колонок и индексы
    def get_header_name_to_idx_maps(self, headers):
        name_to_idx = {}
        idx_to_name = {}
        for i in range(0, len(headers)):
            name_to_idx[headers[i]] = i
            idx_to_name[i] = headers[i]
        return idx_to_name, name_to_idx
    # Получение колонки по имени атрибута
    def get(self, attribute):
        num = self.data['name_to_idx'][attribute]
        col = []
        for row in self.data['rows']:
            col.append(row[num])
        return col
    def delete(self, attribute):
        self.columns_to_project.remove(attribute)
        self.init_data()

def main():
    data_filename = 'horse-colic.data'
    headers_filename = 'horse-colic-headers.json'
    columns_to_project = ['surgery?', 'nasogastric reflux',
                          'abdominocentesis appearance', 'outcome', 'surgical lesion?', 'cp_data']
    target_attribute = 'surgery?'
    parser = Parser(data_filename, headers_filename, columns_to_project)

    Y = parser.get(target_attribute)
    parser.delete(target_attribute)
    data = parser.data
    X = data['rows']

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, Y)
    dot_data = tree.export_graphviz(clf, out_file=None, feature_names=data['header'],
                                    filled=True, rounded=True, special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render("horse-colic")

if __name__ == "__main__": main()
