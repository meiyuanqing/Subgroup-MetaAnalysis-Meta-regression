#!/usr/bin/env python
# encoding:utf-8
"""
Author : Yuanqing Mei
Date : 2021/5/3
Update: 2021/8/26
Time: 9:40
File: sub_AUC.py
HomePage : http://github.com/yuanqingmei
Email : dg1533019@smail.nju.edu.cn

Notes:
    (1) Compute the value of AUC, by Wilcoxon rank sum method using mannwhitneyu() from scipy.stats.mannwhitneyu,
        after comparing scipy.stats.wilcoxon, scipy.stats.ranksums, and scipy.stats.mannwhitneyu.

Outputs:
(1) 第一个文件：对每个metric，列出它预测Buggy的auc及方差，和样本规模，可以供Comprehensive meta-analysis软件使用，进行auc的元分析;
(2) 第二个文件：输出每个metric的auc的子组元分析（subgroup analysis）结果，注意其中使用了该metric与Buggy的Pearson相关系数的方向来
              确定是AUC还是1-AUC;
(3) 第三个文件：元回归的结果;
(4) 第四个文件：子组元分析的方差分析检验;
(5) 第五个文件：子组元分析的异质性检验;
(6) 第六个文件：每个metrics的特征，为画箱线图准备的.

References:
[1] M.Borenstein, L.V. Hedges, J.P.T. Higgins, H.R. Rothstein. Introduction to meta-analysis, John Wiley & Sons, 2009;
[2] Comprehensive meta-analysis. http://www.meta-analysis.com.
[3] Feng Zhang,Audris Mockus,Iman Keivanloo,Ying Zou  Towards building a universal defect prediction model with rank
    transformed predictors Springer Science+Business Media , New York 2015.
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

    # Compute the AUC between metricData and metricData;
    # When metricData is negatively related to metricData, the value of AUC is actually identical to 1 - AUC.
    def auc_man(metricData, defectData):
        normalData = []
        abnormalData = []
        # print("len(metricData) = ", len(metricData))
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

        combined = np.array([metricData, defectData])
        corr_pd = pd.DataFrame(combined.T, columns=['metric', 'defect'])
        corr_matrix = corr_pd.corr(method='spearman')
        corr = corr_matrix.loc["metric", "defect"]

        if corr < 0:
            AUC = 1 - AUC
        print("repr of metricData is ", repr(metricData))
        print("repr of defectData is ", type(defectData))
        print("corr_pd is ", corr_pd)
        print("corr_matrix is ", corr_matrix)
        print("corr is ", corr)
        print("type of corr_matrix is ", type(corr_matrix))
        # print("RankSum = ", RankSum)
        # print("U1 = ", U1)
        # print("U1 = ", U1[0])
        # print("n0 = ", n0)
        # print("n1 = ", n1)
        # print("metricData = ", metricData)
        # print("defectData = ", defectData)
        # print("normalData = ", normalData)
        # print("abnormalData = ", abnormalData)
        # print("AUC = ", AUC)

        if n0 * n1 < 1:
            return 0, 0, 0, 0, 0
        else:
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

    mmm = [1, 2, 2, 1, 0, 2, 1, 7, 4, 5, 6, 3, 6, 4, 4]
    mm = [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0]
    cc = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    print(auc_man(mm, cc))
    auc_value = roc_auc_score(mm, cc)
    print(auc_value)

if __name__ == '__main__':
    s_time = time.time()
    sub_AUC()
    e_time = time.time()
    execution_time = e_time - s_time
    print("The __name__ is ", __name__, ". This is end of sub_AUC.py!",
          "\nThe execution time of sub_AUC.py script is ", execution_time)
