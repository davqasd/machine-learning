#! /usr/bin/env python
# -*- coding: utf-8 -*-

# python id3.py
import csv
import json
import math # для log
import os # для красивого print

# Нельзя просто убрать все строки с ? - иначе не останется данных, нужно
# очищать данные только в выбранных столбцах одновременно
class Parser:
    def __init__(self, data_filename, headers_filename, columns_to_project):
        f = open(data_filename, "r")
        parsed_file = list(csv.reader(f.readlines(), delimiter = ' '))
        headers_data = open(headers_filename, "r").read()
        headers = []
        data_values = json.loads(headers_data)["headers"]
        for header in data_values:
            headers.append(header.keys()[0])

        data = { 'headers': headers, 'rows': parsed_file }
        idx_to_name, name_to_idx = self.get_header_name_to_idx_maps(headers)

        project_column_ids = [name_to_idx[name] for name in columns_to_project]

        rows = self.create_data_with_filter(parsed_file, project_column_ids, data_values)
        idx_to_name, name_to_idx = self.get_header_name_to_idx_maps(columns_to_project)
        self.data = {
            'header': columns_to_project,
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
                    vals.append(data_values[index].values()[0][int(val) - 1]) # значения с 1, индексация массива с 0
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


def get_uniq_values(data):
    idx_to_name = data['idx_to_name']
    idxs = idx_to_name.keys()

    val_map = {}
    for idx in iter(idxs):
        val_map[idx_to_name[idx]] = set()

    for data_row in data['rows']:
        for idx in idx_to_name.keys():
            att_name = idx_to_name[idx]
            val = data_row[idx]
            if val not in val_map.keys():
                val_map[att_name].add(val)
    print
    print 'get_uniq_values:', val_map
    print
    return val_map

# Подсчёт количества значений каждого класса
# Например в самом начале {'Yes, it had surgery': 63, 'It was treated without surgery': 33}
def get_class_labels(data, target_attribute):
    rows = data['rows']
    col_idx = data['name_to_idx'][target_attribute]
    labels = {}
    for r in rows:
        val = r[col_idx]
        if val in labels:
            labels[val] = labels[val] + 1
        else:
            labels[val] = 1
    return labels


def entropy(n, labels):
    ent = 0
    for label in labels.keys():
        p_x = labels[label] / float(n)
        ent += - p_x * math.log(p_x, 2)
    return ent

# Тупо берёт и для каждого значения из data['rows'][номер значения по group_att]
# добавляет ту строку из data, в которой он находится, то есть идёт разбиение данных по group_att
# Например:
# Ветер | Температура | Можно ли играть в гольф
# +     | cold        | -
# -     | warm        | +
# +     | hot         | +
# group_att = 'Ветер',
# Получаем выборку для +: {'+': {'rows': [['+', 'cold', '-'], ['+', 'hot', '+']]}}
# Получаем выборку для -: {'-': {'rows': [['-', 'warm', '+']}}
def partition_data(data, group_att):
    partitions = {}
    data_rows = data['rows']
    partition_att_idx = data['name_to_idx'][group_att]
    for row in data_rows:
        row_val = row[partition_att_idx]
        if row_val not in partitions.keys():
            partitions[row_val] = {
                'name_to_idx': data['name_to_idx'],
                'idx_to_name': data['idx_to_name'],
                'rows': list()
            }
        partitions[row_val]['rows'].append(row)
    return partitions

# Находим энтропию атрибута, как P(x1)*E(x1) + P(x2)*E(x2), где x1, x2, xn -
# события атрибута splitting_att которые наступят для всех значений target_attribute
def avg_entropy_w_partitions(data, splitting_att, target_attribute):
    # find uniq values of splitting att
    data_rows = data['rows']
    n = len(data_rows)
    partitions = partition_data(data, splitting_att)

    avg_ent = 0

    for partition_key in partitions.keys():
        partitioned_data = partitions[partition_key]
        partition_n = len(partitioned_data['rows'])
        partition_labels = get_class_labels(partitioned_data, target_attribute)
        partition_entropy = entropy(partition_n, partition_labels)
        avg_ent += partition_n / float(n) * partition_entropy

    return avg_ent, partitions


def most_common_label(labels):
    mcl = max(labels, key=lambda k: labels[k])
    return mcl


def id3(data, uniqs, remaining_atts, target_attribute):
    labels = get_class_labels(data, target_attribute)
    node = {}

    # Если не из чего выбирать. То есть если в выборке только один атрибут, например
    # Ветер: есть: 1, нет: 0 => выбираем только значение, когда есть ветер
    if len(labels.keys()) == 1:
        # next(iter(labels.keys())) === labels.keys()[0]
        node['label'] = next(iter(labels.keys()))
        return node
    # Условие выхода из рекурсии, когда все категориальные фичи просмотрены
    if len(remaining_atts) == 0:
        node['label'] = most_common_label(labels)
        return node

    n = len(data['rows'])
    ent = entropy(n, labels) # общая энтропия выборки

    max_info_gain = None
    max_info_gain_att = None
    max_info_gain_partitions = None

    for remaining_att in remaining_atts:
        avg_ent, partitions = avg_entropy_w_partitions(data, remaining_att, target_attribute)
        info_gain = ent - avg_ent
        print "IG for attribute '" + remaining_att + "' = " + str(info_gain)
        if max_info_gain is None or info_gain > max_info_gain:
            max_info_gain = info_gain
            max_info_gain_att = remaining_att
            max_info_gain_partitions = partitions
    print "!!!"

    if max_info_gain is None:
        node['label'] = most_common_label(labels)
        return node

    node['attribute'] = max_info_gain_att
    node['nodes'] = {}

    remaining_atts_for_subtrees = set(remaining_atts)
    # my_set = {1, 2, 3, 4, 5}
    # my_set.discard(2)
    # my_set  # {1, 3, 4, 5}
    remaining_atts_for_subtrees.discard(max_info_gain_att) # важный шаг - исключение выбранного атрибута

    uniq_att_values = uniqs[max_info_gain_att]

    for att_value in uniq_att_values:
        if att_value not in max_info_gain_partitions.keys():
            node['nodes'][att_value] = {'label': most_common_label(labels)}
            continue
        # теперь данные - это ограниченная выборка для данных от атрибута с максимальной IG
        partition = max_info_gain_partitions[att_value]
        node['nodes'][att_value] = id3(partition, uniqs, remaining_atts_for_subtrees, target_attribute)

    return node


def pretty_print_tree(root):
    stack = []
    rules = set()

    def traverse(node, stack, rules):
        if 'label' in node:
            stack.append(' THEN ' + node['label'])
            rules.add(''.join(stack))
            stack.pop()
        elif 'attribute' in node:
            ifnd = 'IF ' if not stack else ' AND '
            stack.append(ifnd + node['attribute'] + ' EQUALS ')
            for subnode_key in node['nodes']:
                stack.append(subnode_key)
                traverse(node['nodes'][subnode_key], stack, rules)
                stack.pop()
            stack.pop()

    traverse(root, stack, rules)
    print(os.linesep.join(rules))
    print root

# TODO: Нормальная отрисовка дерева
def main():
    data_filename = 'horse-colic.data'
    headers_filename = 'horse-colic-headers.json'
    columns_to_project = ['surgery?', 'nasogastric reflux',
                          'abdominocentesis appearance', 'outcome', 'surgical lesion?', 'cp_data']
    parser = Parser(data_filename, headers_filename, columns_to_project)
    data = parser.data
    print 'data: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
    print 'header:', data['header']
    print 'rows:', data['rows']
    print 'name_to_idx:', data['name_to_idx']
    print 'idx_to_name:', data['idx_to_name']
    print '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'

    target_attribute = 'surgery?'

    remaining_attributes = set(data['header'])
    remaining_attributes.remove(target_attribute)
    uniqs = get_uniq_values(data)
    root = id3(data, uniqs, remaining_attributes, target_attribute)
    pretty_print_tree(root)


if __name__ == "__main__": main()
