#!/usr/bin/env python
# encoding:utf-8
"""
Author : Yuanqing Mei
Date : 2021/8/29
Time: 21:10
File: AUC_m.py
HomePage : http://github.com/yuanqingmei
Email : dg1533019@smail.nju.edu.cn

The method of computing AUC value!

"""
import time


def sub_AUC(working_dir="F:\\NJU\\subMeta\\experiments\\preprocess\\PL\\",
            result_dir="F:\\NJU\\subMeta\\experiments\\subgroupPearson\\PearsonEffect\\",
            training_list="List.txt"):
    import os
    import csv
    import scipy
    import numpy as np
    import pandas as pd
    from sklearn.metrics import recall_score, precision_score, f1_score, roc_curve, auc, roc_auc_score, confusion_matrix

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 5000)

    result_directory = result_dir
    os.chdir(working_dir)

    def auc_man(metricData, defectData):
        normalData = []
        abnormalData = []
        # compute the Spearman coefficient between metricData and defectData
        combined = np.array([metricData, defectData])
        corr_pd = pd.DataFrame(combined.T, columns=['metric', 'defect'])
        corr_matrix = corr_pd.corr(method='spearman')
        corr = corr_matrix.loc["metric", "defect"]

        for i in range(len(metricData)):
            # print(i, defectData[i])
            if metricData[i] == "und":
                continue
            if defectData[i] > 0:
                abnormalData.append(metricData[i])
            else:
                normalData.append(metricData[i])

        n0 = len(normalData)
        n1 = len(abnormalData)

        # When all the modules in a system are defective or defect-free (i.e., the number of defects is all zero or
        # all non-zero), no metric in the system can distinguish the defective or non-defective classes,
        # and the value of AUC is 0.5 (i.e., the same as that predicted by the random model).
        if n0 * n1 < 1:
            return 0.5, 0, corr, (n0 + n1), n1

        U1 = scipy.stats.mannwhitneyu(normalData, abnormalData)
        # 2021/8/28: The formula of U1 is U1 = R1 - n1 * (n1 + 1) / 2 in scipy.stats.mannwhitneyu, which is identical to
        # U2 actually. The correct formula of U1 is U1 = n1 * n2 + n1*(n1+1)/2 - R1. As it not revised in scipy 1.7.1,
        # we use the alternative value as U1, namely len(normalData) * len(abnormalData) - U1[0].
        # RankSum = n0 * n1 + 0.5 * n0 * (n0 + 1) - U1[0]
        RankSum = n0 * n1 + 0.5 * n0 * (n0 + 1) - (n0 * n1 - U1[0])

        MW_U = RankSum - 0.5 * n0 * (n0 + 1)
        AUC = (n0 * n1 - MW_U) / (n0 * n1)

        Q1 = AUC / (2 - AUC)
        Q2 = 2 * AUC * AUC / (1 + AUC)

        variance = (AUC * (1 - AUC) + (n1 - 1) * (Q1 - AUC * AUC) + (n0 - 1) * (Q2 - AUC * AUC)) / (n0 * n1)

        if corr < 0:
            AUC = 1 - AUC

        return AUC, variance, corr, (n0 + n1), n1

    x = [9, 5, 8, 7, 10, 6, 7]
    y = [7, 4, 5, 6, 3, 6, 4, 4]
    males = [19, 22, 16, 29, 24]
    females = [20, 11, 17, 12]
    print(scipy.stats.ranksums(x, y))
    print(scipy.stats.mannwhitneyu(x, y))
    print(scipy.stats.mannwhitneyu(y, x))
    U1 = scipy.stats.mannwhitneyu(x, y)
    print(type(U1))
    print(repr(U1))
    print(U1[0])
    R = len(x) * len(y) + 0.5 * len(x) * (len(x) + 1) - U1[0]
    print("R = ", R)
    print(scipy.stats.mannwhitneyu(males, females))
    print(scipy.stats.mannwhitneyu(females, males))

    # mm = [9, 5, 8, 7, 10, 6, 7, 7, 4, 5, 6, 3, 6, 4, 4]
    mm = [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0]
    cc = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    print(auc_man(mm, cc))
    auc_value = roc_auc_score(mm, cc)
    print(auc_value)

    x = [9, 5, 8, 7, 10, 6, 7]
    y = [7, 4, 5, 6, 3, 6, 4, 4]
    males = [19, 22, 16, 29, 24]
    females = [20, 11, 17, 12]
    print(scipy.stats.ranksums(x, y))
    print(scipy.stats.mannwhitneyu(x, y))
    print(scipy.stats.mannwhitneyu(y, x))
    U1 = scipy.stats.mannwhitneyu(x, y)
    print(type(U1))
    print(repr(U1))
    print(U1[0])
    R = len(x) * len(y) + 0.5 * len(x) * (len(x) + 1) - U1[0]
    print("R = ", R)
    print(scipy.stats.mannwhitneyu(males, females))
    print(scipy.stats.mannwhitneyu(females, males))

    mmm = [1, 2, 2, 1, 0, 2, 1, 7, 4, 5, 6, 3, 6, 4, 4]
    mm = [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0]
    ccc = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    mmmm = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    cccc = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ccccc = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # cc = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    print(auc_man(cccc, mmmm))
    auc_value = roc_auc_score(mmmm, mmmm)
    # auc_value = roc_auc_score(mm, cccc)
    print(auc_value)

if __name__ == '__main__':
    s_time = time.time()
    sub_AUC()
    e_time = time.time()
    execution_time = e_time - s_time
    print("The __name__ is ", __name__, ". This is end of sub_AUC.py!",
          "\nThe execution time of sub_AUC.py script is ", execution_time)