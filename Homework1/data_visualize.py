# -*- coding=UTF-8 -*-
import data_abstract
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import itertools
import json
import copy

from multiprocessing import Process

# 盒图
def draw_box(column, values_clean, loc):
    plt.figure(figsize=(2.8,2))
    # 离散点 (图标样式,图标颜色,大小,..)
    fp = {'marker': "o", 'markerfacecolor': 'blue', 'markersize': 5, 'linestyle': 'none'}
    plt.title("Box:" + str(column))
    plt.boxplot(values_clean, flierprops=fp)
    plt.savefig(loc+'box_'+column+'.png')
    # plt.show()
    plt.close()
    pass


def draw_hist(column, vc, loc):
    plt.figure(figsize=(2.8, 2))
    plt.title("Hist:" + str(column))
    plt.hist(vc, bins=20)
    plt.savefig(loc+'hist_'+column+'.png')
    plt.close()
    pass


def draw_qq_norm(column, vc, loc):
    plt.figure(figsize=(2.8, 2))
    stats.probplot(vc, dist="norm", plot=plt)
    plt.title("Q-Q:" + str(column))
    plt.savefig(loc+'qq_'+column+'.png')
    # plt.show()
    plt.close()
    pass

# qq图检测两属性间的相关度
def draw_qq_double(csv_file, double_column):
    data = csv_file[list(double_column)].dropna()
    x = data[double_column[0]].values
    y = data[double_column[1]].values

    plt.figure(figsize=(2.8,2))
    plt.title(double_column[0] + "_" + double_column[1])
    plt.plot(x, y, 'ro')
    plt.savefig('diagram/diagram_compare/'+double_column[0]+"_"+double_column[1]+'.png')
    plt.show()
    plt.close()


# 去除缺失值 绘图函数
def complete_dropna(csv_file, column):
    loc = "diagram/diagram_after_completion/type1_"
    values_dropna = csv_file[column].dropna().values
    draw_hist(column, values_dropna, loc)
    draw_qq_norm(column, values_dropna, loc)
    draw_box(column, values_dropna, loc)
    pass

# 用最高频率值来填补缺失值 绘图函数
def complete_fre_attr(csv_file, column):
    value_count = csv_file[column].dropna().value_counts()
    max_fre_value = value_count.index[0]
    data = csv_file[column]
    miss_index = data[data.isnull()].index
    complete_data = data.copy()
    for i in miss_index:
        complete_data[i] = max_fre_value

    loc = "diagram/diagram_after_completion/type2_"
    draw_hist(column, complete_data, loc)
    draw_qq_norm(column, complete_data, loc)
    draw_box(column, complete_data, loc)

# 通过属性的相关关系来填补缺失值 绘图函数
def complete_rel_attr(csv_file, double_column):
    target_data = csv_file[double_column[0]]
    source_data = csv_file[double_column[1]]
    flag1 = target_data.isnull().values
    flag2 = source_data.isnull().values
    complete_data = target_data.copy()
    for index, value in target_data.iteritems():
        if flag1[index] == True and flag2[index] == False:

            complete_data[index] = 1 - source_data[index]

    values_clean = list(complete_data.dropna().values)
    for value, count in complete_data.value_counts().iteritems():
        if count == 1:
            values_clean.remove(value)

    loc = "diagram/diagram_after_completion/type3_"
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

# 通过数据对象之间的相似性来填补缺失值 绘图函数
def complete_smi_attr(csv_file, column, numeric_attr):
    data = csv_file[column].copy()
    for index, value in data.iteritems():
        if value == np.NaN:
            data[index] = data[find_dis_value(csv_file, index, column, numeric_attr)]
    loc = "diagram/diagram_after_completion/type4_"
    draw_hist(column, data.dropna().values, loc)
    draw_qq_norm(column, data.dropna().values, loc)
    draw_box(column, data.dropna().values, loc)


# 画图
def draw_numeric(csv_file, numeric_attr):
    for column in numeric_attr:
        print("clean_before")
        values_dropna, values_clean, vc = data_abstract.clean_data(csv_file, column, 0.05)
        print("clean_over")
        loc = 'diagram/'
        draw_hist(column, vc, loc)
        print("hist_over")
        draw_qq_norm(column, vc, loc)
        print("qq_over")
        draw_box(column, values_clean, loc)

