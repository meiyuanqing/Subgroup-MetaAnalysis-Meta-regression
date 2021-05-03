#!/usr/bin/env python
# encoding:utf-8
"""
Author : Yuanqing Mei
Date : 2021/5/3
Time: 9:40
File: sub_pearson.py
HomePage : http://github.com/yuanqingmei
Email : dg1533019@smail.nju.edu.cn

Gets the effective size and it's variance for Pearson, then for meta-analysis.
主要步骤：
    (1)计算Spearman相关系数Spearman_value；
    (2)由于Pearson相关系数需要度量与缺陷变量满足正态分布，
       计算近似Pearson相关系数：Pearson_value = 2 * np.sin(np.pi * Spearman_value / 6)
    (3)通过Fisher变换：Fisher_Z = 0.5 * np.log((1 + Pearson_value) / (1 - Pearson_value))
    (4)计算Fisher_Z的方差： Fisher_Z_variance = 1 / (Sample_size - 3), Sample_size为第i系统上样本数；
    (5)然后对Fisher_Z做随机效应元分析，最后通过Fisher反向变换，得出Pearson的元分析值，其符号为方向，即正号为正相关，负号为负相关。

(1) 第一个文件：对每个metric，列出它预测Buggy的auc及方差，和样本规模，可以供Comprehensive meta-analysis软件使用，进行auc的元分析
(2) 第二个文件：输出每个metric的auc的子组元分析（subgroup analysis）结果，注意其中使用了该metric与Buggy的Pearson相关系数的方向
              来确定是AUC还是1-AUC
(3) 第三个文件：元回归的结果
(4) 第四个文件：子组元分析的方差分析检验
(5) 第五个文件：子组元分析的异质性检验
(6) 第六个文件：每个metrics的特征，为画箱线图准备的

"""
import time


def sub_pearson(working_dir="F:\\NJU\\subMeta\\experiments\\preprocess\\PL\\c\\",
                result_dir="F:\\NJU\\subMeta\\experiments\\subgroupPearson\\PearsonEffect\\c\\",
                training_list="c_List.txt"):
    import os
    import csv
    import numpy as np
    import pandas as pd

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 5000)

    working_directory = working_dir
    result_directory = result_dir
    os.chdir(working_directory)

    with open(working_dir + training_list) as l:
        lines = l.readlines()

    for line in lines:
        file = line.replace("\n", "")
        print('the file is ', file)

        # 分别处理每一个项目: f1取出要被处理的项目;
        #                  f2:用于存储每一个项目的Spearman,pearson,FisherZ和variance
        #                  deletedList: 用于存储项目中某个度量样本数小于3，和pearson等于1的度量。
        with open(working_directory + file, 'r', encoding="ISO-8859-1") as f1, \
                open(result_directory + "Pearson_effects.csv", 'a+', encoding="utf-8", newline='') as f2, \
                open(result_directory + "Pearson_effects_deletedList.csv", 'a+', encoding="utf-8", newline='') \
                        as deletedList:

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

            df = pd.read_csv(file)
            # drop all rows that have any NaN values,删除表中含有任何NaN的行,并重新设置行号
            df = df.dropna(axis=0, how='any', inplace=False).reset_index(drop=True)

            if os.path.getsize(result_directory + "Pearson_effects.csv") == 0:
                writer.writerow(["fileName", "metric", "Sample_size", "Spearman_metric_bug", "Pearson_metric_bug",
                                 "Fisher_Z", "Fisher_Z_variance"])

            if os.path.getsize(result_directory + "Pearson_effects_deletedList.csv") == 0:
                writer_deletedList.writerow(["fileName", "metric", "Sample_size", "Spearman_metric_bug",
                                             "Pearson_metric_bug", "Fisher_Z", "Fisher_Z_variance"])

            for metric in metric_data:
                print("the current file is ", file)
                print("the current metric is ", metric)

                # 判断每个度量与bug之间的关系,因为该关系会影响到断点回归时,相关系数大于零,则LATE估计值大于零,反之,则LATE估计值小于零
                Spearman_metric_bug = df.loc[:, [metric, 'bug']].corr('spearman')

                Spearman_value = Spearman_metric_bug[metric][1]
                # print("the Spearman_value is ", Spearman_value, type(Spearman_value), repr(Spearman_value))
                # print("the boolean is ", np.isnan(Spearman_value))
                Pearson_value = 2 * np.sin(np.pi * Spearman_value / 6)

                Sample_size = len(df[metric])

                if (Sample_size <= 3) or (Pearson_value == 1) or np.isnan(Spearman_value):
                    Fisher_Z = 0
                    Fisher_Z_variance = 0
                    writer_deletedList.writerow([file, metric, Sample_size, Spearman_value, Pearson_value, Fisher_Z,
                                                 Fisher_Z_variance])
                else:
                    Fisher_Z = 0.5 * np.log((1 + Pearson_value) / (1 - Pearson_value))
                    Fisher_Z_variance = 1 / (Sample_size - 3)
                    writer.writerow([file, metric, Sample_size, Spearman_value, Pearson_value, Fisher_Z,
                                     Fisher_Z_variance])


if __name__ == '__main__':
    s_time = time.time()

    working_dir = "F:\\NJU\\subMeta\\experiments\\preprocess\\PL\\cpp\\"
    result_dir = "F:\\NJU\\subMeta\\experiments\\subgroupPearson\\PearsonEffect\\cpp\\"
    training_list = "cpp_List.txt"

    sub_pearson(working_dir, result_dir, training_list)
    e_time = time.time()
    execution_time = e_time - s_time
    print("The __name__ is ", __name__, ". This is end of sub_pearson.py!",
          "\nThe execution time of sub_pearson.py script is ", execution_time)
