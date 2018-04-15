# -*- coding=UTF-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import itertools
import json
import copy
import data_abstract
import data_visualize

from multiprocessing import Process


if __name__ == "__main__":
    homework_file1 = r"E:\datamining\Building_Permits.csv"
    homework_file2 = r"E:\datamining\NFL Play by Play 2009-2017.csv"

    # ----------数据集1---------------
    # 人工选择所有数值属性
    numeric_attr1 = ['Number of Existing Stories', 'Number of Proposed Stories', 'Estimated Cost', 'Revised Cost',
                    'Existing Units', 'Proposed Units']
    csv_file1 = data_abstract.open_csv(homework_file1)

    data_abstract.nominal_statistic(csv_file1, numeric_attr1, "Building_Permits")
    data_abstract.numeric_statistic(csv_file1, numeric_attr1, "Building_Permits")
    data_visualize.draw_numeric(csv_file1, numeric_attr1)

    # 绘制两个属性的qq图，判断相关性
    for double_column in itertools.combinations(numeric_attr1, 2):
        data_visualize.draw_qq_double(csv_file1, double_column)

    # 只展示Estimated Cost属性填补后的效果
    data_visualize.complete_dropna(csv_file1, 'Estimated Cost')
    data_visualize.complete_fre_attr(csv_file1, 'Estimated Cost')
    data_visualize.complete_rel_attr(csv_file1, ['Estimated Cost', 'Revised Cost'])
    data_visualize.complete_smi_attr(csv_file1, 'Estimated Cost', numeric_attr1)


    # ----------数据集2---------------
    # csv_file2 = open_csv(homework_file2)
    # 找出所有数值属性
    # numeric_attr2 = select_attr(csv_file2)

    # numeric_attr2 = ['Away_WP_pre', 'FieldGoalDistance', 'Field_Goal_Prob', 'Home_WP_pre', 'Opp_Field_Goal_Prob', 'Touchdown_Prob', 'yrdline100', 'yrdln']
    #data_abstract.nominal_statistic(csv_file2, numeric_attr2, "NFL Play by Play 2009-2017 (v4)")
    #data_abstract.numeric_statistic(csv_file2, numeric_attr2, "NFL Play by Play 2009-2017 (v4)")
    #data_visualize.draw_numeric(csv_file2, numeric_attr2)
    # def child(csv_file, column):
    #     draw_numeric(csv_file, [column])
    #
    #
    # childs = []
    # for column in numeric_attr2:
    #     p = Process(target=child, args=(csv_file2, column))
    #     p.start()
    #     childs.append(p)
    #
    # for child_p in childs:
    #     child_p.join()

    # for double_column in itertools.combinations(numeric_attr2, 2):
    #     draw_qq_double(csv_file2, double_column)

    # complete_dropna(csv_file2, 'Home_WP_pre')
    # complete_fre_attr(csv_file2, 'Home_WP_pre')
    # complete_rel_attr(csv_file2, ['Home_WP_pre', 'Away_WP_pre'])
    # complete_smi_attr(csv_file2, 'Away_WP_pre', numeric_attr2)
