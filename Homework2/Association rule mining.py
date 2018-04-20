# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from itertools import chain, combinations
from collections import defaultdict


def open_file(file_to_open):
    _csv = pd.read_csv(file_to_open, low_memory=False)
    return _csv


def preprocess(file):
    column_suitable = []
    for column in file.columns:
        if 0 < file[column].value_counts().__len__() < 20:
            print column
            column_suitable.append(column)
    column_split = []
    for column in column_suitable:
        words = column.split(' ')
        name = ''
        for word in words:
            name += word[0]
            column_split.append(name)
# 截取最终用于处理的的数据集
    data = file[column_suitable]
    trans_dict = {}
    record_num = data.index.__len__()
    for column in column_suitable:
        new_line = [""]*record_num
        for index in data.index:
            item = data[column][index]
            try:
                if np.isnan(item):
                    new_line[index] = ""
                else:
                    new_line[index] = column_split[column_suitable.index(column)] + "_"+ str(item)
            except BaseException as e:
                new_line[index] = column_split[column_suitable.index(column)] + "_" + str(item)
        trans_dict[column] = new_line
    csv_new = pd.DataFrame(trans_dict)
    csv_new.to_csv('preprocessed_data.csv', index=False, header=False)


# 求满足最小支持度的子集
def min_support(item_set, transaction_list, min_sup, freqs):
    _items = set()
    local_set = defaultdict(int)

    for item in item_set:
        for transaction in transaction_list:
            if item.issubset(transaction):
                freqs[item] += 1
                local_set[item] += 1
    for item, count in local_set.items():
        support = float(count)/len(transaction_list)
        if support >= min_sup:
            _items.add(item)
    return _items


def joinSet(item_set, length):
    return set([i.union(j) for i in item_set for j in item_set if len(i.union(j)) == length])


def getItemSetTransactionList(data_iterator):
    transactionList = list()
    item_set = set()
    for record in data_iterator:
        transaction = frozenset(record)
        transactionList.append(transaction)
        for item in transaction:
            item_set.add(frozenset([item]))
    return item_set, transactionList


# 返回值：频繁项集和关联规则
def algo_apriori(data_iter, min_sup, min_confidence):
    item_set, transaction_list = getItemSetTransactionList(data_iter)
    # 所有项的频数 频繁项和非频繁项，1项和K项
    freqs = defaultdict(int)
    # 存储各元频繁项集合 key=K，value=K项频繁项集合
    large_set = dict()
    one_con_set = min_support(item_set, transaction_list, min_sup, freqs)
    current_l_set = one_con_set
    k = 2
    # 递归逐层求解
    while(current_l_set != set([])):
        large_set[k-1] = current_l_set
        current_l_set = joinSet(current_l_set, k)
        current_c_set = min_support(current_l_set, transaction_list, min_sup, freqs)
        current_l_set = current_c_set
        k = k + 1

    def get_support(item):
            return float(freqs[item])/len(transaction_list)

    get_items = []
    for key, value in large_set.items():
        Items.extend([(tuple(item), get_support(item))
                           for item in value])

    get_rules = []
    for key, value in large_set.items()[1:]:
        for item in value:
            _subsets = map(frozenset, [x for x in subsets(item)])
            for element in _subsets:
                remain = item.difference(element)
                if len(remain) > 0:
                    confidence = get_support(item)/get_support(element)
                    if confidence >= min_confidence:
                        get_rules.append(((tuple(element), tuple(remain)), get_support(item),
                                           confidence, confidence/get_support(remain)))
    return get_items, get_rules


def results(items, rules):
    fre_file = open(r'frequency_results.txt', 'w')
    ass_rules_by_confidence = open(r'confidence_results.txt', 'w')
    ass_rules_by_lift = open(r'lift_results.txt', 'w')
    # 按confidence值由大到小排序
    ass_rules_by_confidence.write('rule\tsupport\tconfidence\tlift\n')
    for rule, support, confidence, lift in sorted(rules, key=lambda (rule, suppport, confidence, lift): confidence, reverse=True):
        pre, post = rule
        ass_rules_by_confidence.write("%s -> %s\t%.3f\t%.3f\t%.3f\n" % (str(pre), str(post), support, confidence, lift))
    ass_rules_by_confidence.close()
    fre_file.write('frequent_items\tsupport\n')
    # 按lift值由大到小排列
    ass_rules_by_lift.write('rule\tsupport\tconfidence\tlift\n')
    for rule, support, confidence, lift in sorted(rules, key=lambda (rule, suppport, confidence, lift): lift, reverse=True):
        pre, post = rule
        ass_rules_by_lift.write("%s -> %s\t%.3f\t%.3f\t%.3f\n" % (str(pre), str(post), support, confidence, lift))
    ass_rules_by_lift.close()
    # 按support值由大到小排列
    for item, support in sorted(items, key=lambda (item, support): support, reverse=True):
        fre_file.write("%s\t%.3f\n" % (str(item), support))
    fre_file.close()


def dataFromFile(fname):
        file_iter = open(fname, 'rU')
        for line in file_iter:
                line = line.strip().rstrip(',')
                shopping_basket = line.split(',')
                while '' in shopping_basket:
                    shopping_basket.remove('')
                record = frozenset(shopping_basket)
                yield record


# 返回数组arr的非空子集
def subsets(arr):
    return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])


if __name__ == "__main__":

    homework_file1 = r"E:\datamining\Building_Permits.csv"
    csv_file = open_file(homework_file1)
    preprocess(csv_file)
    inFile = dataFromFile('preprocessed_data.csv')
    min_support = 0.3
    min_confidence = 0.7
    items, rules = algo_apriori(inFile, min_support, min_confidence)
    results(items, rules)
