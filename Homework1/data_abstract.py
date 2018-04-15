# -*- coding=UTF-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import itertools
import json
import copy

from multiprocessing import Process


def open_csv(file):
    csv_file = pd.read_csv(file, low_memory=False)
    return csv_file


def nominal_statistic(csv_file, numeric_attr, name):
    result_dict = {}
    for column in csv_file.columns:
        if column not in numeric_attr:
            result_dict[column] = csv_file[column].value_counts().to_dict()
    json.dump(str(result_dict), open(r'process_result/'+name+'_frequency.json', 'w'))


def numeric_statistic(csv_file, numeric_attr, name):
    result_dict = {}
    for column in numeric_attr:
        column_series = copy.copy(csv_file[column])
        clean_series = column_series.dropna()

        num_of_NaN = column_series.__len__() - clean_series.__len__()

        clean_list = clean_series.values.tolist()

        clean_list.sort()
        len = clean_list.__len__()
        max_value = clean_list[-1]
        min_value = clean_list[0]
        sum_value = sum(clean_list)
        mean_value = sum_value / clean_list.__len__()

        Q1 = clean_list[int((len + 1) * 0.25)]
        Q2 = clean_list[int((len + 1) * 0.5)]
        Q3 = clean_list[int((len + 1) * 0.75)]

        result = [max_value, min_value, mean_value, Q2, [Q1, Q2, Q3], num_of_NaN]
        result_dict[column] = result
    json.dump(result_dict, open('process_result/'+name+'_numeric_result.json', 'w'))


def clean_data(csv_file, column, percent):
    # 去除缺失值
    values_dropna = csv_file[column].dropna().values
    values_count = csv_file[column].dropna().value_counts()
    values_clean = list(values_dropna)

    # 去除频率为1的值
    # for value, count in values_count.iteritems():
    #     if count == 1:
    #         values_clean.remove(value)

    # 为加快速度，对所有取值种类的频数-1，近似等效于去除频率为1的值
    for item in values_count.index:
        values_clean.remove(item)

    values_clean.sort()
    len = values_clean.__len__()

    # 按percent比例截尾
    vc = values_clean[int(len * percent):int(len * (1 - percent))]

    return values_dropna, values_clean, vc


    # 去除频率为1的值
    for value, count in complete_data.value_counts().iteritems():
        if count == 1:
            values_clean.remove(value)

    loc = "graph/complete/type3_"
    draw_hist(double_column[0], values_clean, loc)
    draw_qq_norm(double_column[0], values_clean, loc)
    draw_box(double_column[0], values_clean, loc)

# 查找两个对象间相异度最小的 指定的 column值
def find_dis_value(csv_file, pos, column, numeric_attr):

    def dis_objs(tar_obj, sou_obj):
        dis_value = 0
        count = 0
        for column in tar_obj.index:
            if tar_obj[column] != np.NaN and sou_obj[column] != np.NaN:
                if column in numeric_attr:
                        values_sort = csv_file[column].dropna().values.sort()
                        denominator = values_sort[-1] - values_sort[0]
                        dis_value += abs(tar_obj[column] - sou_obj[column])/denominator
                        count += 1

                elif tar_obj[column] == sou_obj[column]:
                    dis_value += 1
                count += 1
            else:
                continue
        return dis_value/count

    mindis = 9999
    result_pos = -1
    target_obj = csv_file.ix[pos]
    for index in csv_file.index:
        if index == pos:
            continue
        source_obj = csv_file.ix(index)
        tmp = dis_objs(target_obj, source_obj)
        if tmp < mindis:
            result_pos = index
    return result_pos


def select_attr(csv_file):
    nominal_attr= []
    columns = csv_file.columns.values
    for column in columns:
        values = csv_file[column].dropna().values
        value_count = csv_file[column].dropna().value_counts()
        if value_count.__len__()<10:
            nominal_attr.append(column)
            continue
        for value in values[:10]:
            try:
                if float(value) > 9999:
                    nominal_attr.append(column)
                    break
            except BaseException as e:
                nominal_attr.append(column)
                break

    return list(set(columns)-set(nominal_attr))


