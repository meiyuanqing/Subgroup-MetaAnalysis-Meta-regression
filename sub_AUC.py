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
            result_dir="F:\\NJU\\subMeta\\experiments\\subgroupMetaAnalysis\\"):
    import os
    import csv
    import scipy
    import numpy as np
    import pandas as pd
    from sklearn.metrics import recall_score, precision_score, f1_score, roc_curve, auc, roc_auc_score, confusion_matrix

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 5000)

    working_directory = working_dir
    result_directory = result_dir
    os.chdir(working_dir)

    # Compute the AUC between metricData and metricData;
    # When metricData is negatively related to metricData, the value of AUC is actually identical to 1 - AUC.
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

    PLs = ["cpp", "cs", "java", "c", "pascal"]

    for PL in PLs:
        print("This is ", PL, " studies!")
        with open(working_directory + PL + "\\" + PL + "_List.txt") as l:
            lines = l.readlines()

        for line in lines:
            file = line.replace("\n", "")
            print('the file is ', file)

            with open(working_directory + PL + "\\" + file, 'r', encoding="ISO-8859-1") as f1, \
                    open(result_directory + "AUC_MetaAnalysis_Data.csv", 'a+', encoding="utf-8", newline='') as f2, \
                    open(result_directory + "AUC_MA_Data_deleted.csv", 'a+', encoding="utf-8",
                         newline='') as deletedList:

                reader = csv.reader(f1)
                writer = csv.writer(f2)
                writer_deletedList = csv.writer(deletedList)
                # receives the first line of a file and convert to dict generator
                fieldnames = next(reader)
                # exclude the non metric fields
                non_metric = ["Host", "Vcs", "Project", "File", "Buggy", "PL", "IssueTracking",
                              "TLOC", "TNF", "TNC", "TND", "bug"]

                # metric_data stores the metric fields (102 items)
                def fun_1(m):
                    return m if m not in non_metric else None

                metric_data = filter(fun_1, fieldnames)

                df = pd.read_csv(working_directory + PL + "\\" + file)
                # drop all rows that have any NaN values,删除表中含有任何NaN的行,并重新设置行号
                df = df.dropna(axis=0, how='any', inplace=False).reset_index(drop=True)

                if os.path.getsize(result_directory + "AUC_MetaAnalysis_Data.csv") == 0:
                    writer.writerow(["fileName", "metric", "Sample_size", "Spearman_metric_bug", "Pearson_metric_bug",
                                     "Fisher_Z", "Fisher_Z_variance", "subGroup", "AUC", "Variance", "NumberOfBug",
                                     "PercentOfBug", "LogOfBug"])

                if os.path.getsize(result_directory + "AUC_MA_Data_deleted.csv") == 0:
                    writer_deletedList.writerow(["fileName", "metric", "Sample_size", "Spearman_metric_bug",
                                                 "Pearson_metric_bug", "Fisher_Z", "Fisher_Z_variance", "subGroup",
                                                 "AUC", "Variance", "NumberOfBug", "PercentOfBug", "LogOfBug"])

                for metric in metric_data:
                    print("the current file is ", file, "the current metric is ", metric)

                    Spearman_metric_bug = df.loc[:, [metric, 'bug']].corr('spearman')

                    Spearman_value = Spearman_metric_bug[metric][1]
                    # print("the Spearman_value is ", Spearman_value, type(Spearman_value), repr(Spearman_value))
                    # print("the boolean is ", np.isnan(Spearman_value))
                    Pearson_value = 2 * np.sin(np.pi * Spearman_value / 6)

                    Sample_size = len(df[metric])
                    print("The size of file is ", Sample_size)
                    AUC = auc_man(df.loc[:, metric], df.loc[:, 'bug'])

                    print("The AUC is ", AUC)
                    AUC_value = AUC[0]
                    AUC_variance = AUC[1]
                    NumberOfBug = AUC[4]
                    PercentOfBug = AUC[4] / AUC[3]
                    LogOfBug = np.log(1 + AUC[4])

                    if (Sample_size <= 3) or (Pearson_value == 1) or np.isnan(Spearman_value):
                        Fisher_Z = 0
                        Fisher_Z_variance = 0
                        writer_deletedList.writerow([file, metric, Sample_size, Spearman_value, Pearson_value, Fisher_Z,
                                                     Fisher_Z_variance, PL, AUC_value, AUC_variance, NumberOfBug,
                                                     PercentOfBug, LogOfBug])
                    else:
                        Fisher_Z = 0.5 * np.log((1 + Pearson_value) / (1 - Pearson_value))
                        Fisher_Z_variance = 1 / (Sample_size - 3)
                        writer.writerow([file, metric, Sample_size, Spearman_value, Pearson_value, Fisher_Z,
                                         Fisher_Z_variance, PL, AUC_value, AUC_variance, NumberOfBug, PercentOfBug,
                                         LogOfBug])
                    break


if __name__ == '__main__':
    s_time = time.time()
    sub_AUC()
    e_time = time.time()
    execution_time = e_time - s_time
    print("The __name__ is ", __name__, ". This is end of sub_AUC.py!",
          "\nThe execution time of sub_AUC.py script is ", execution_time)
