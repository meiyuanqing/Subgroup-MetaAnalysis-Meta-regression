#!/usr/bin/env python
# encoding:utf-8
"""
Author : Yuanqing Mei
Date : 2021/4/28
Time: 23:12
File: preprocess_data.py
HomePage : http://github.com/yuanqingmei
Email : njumyq@outlook.com
Function:

Preprocesses to divide the rawMetrics.csv into five language file.

RawMetrics.csv file comes from the reference:
    《Towards building a universal defect prediction model with rank transformed predictors》

"""
import time


def preprocess_data(pre_dir="F:\\NJU\\subMeta\\experiments\\preprocess\\"):
    import os
    import pandas as pd

    os.chdir(pre_dir)
    print(os.getcwd())

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 5000)

    # read_csv(path, keep_default_na=False, na_values=[""])  只有一个空字段将被识别为NaN
    df = pd.read_csv(pre_dir + "rawMetrics.csv", keep_default_na=False, na_values=[""])

    PL_names = sorted(set(df.PL.values.tolist()))
    print("the metric_names are ", df.columns.values.tolist())
    print("the metric_names' len is ", len(df.columns.values.tolist()))
    print("the PL_names are ", PL_names)
    print("the len of PL_names is ", len(PL_names))

    # for column in df.columns.values.tolist():
    #     print(column)
    for PL_name in PL_names:
        print("The current PL is ", PL_name)
        PL_df = df[df["PL"] == PL_name].loc[:, :]
        PL_df.to_csv(pre_dir + PL_name + ".csv", encoding="ISO-8859-1", index=False, mode='a')


if __name__ == '__main__':
    s_time = time.time()
    preprocess_data()
    e_time = time.time()
    execution_time = e_time - s_time
    print("The __name__ is ", __name__, ". This is end of preprocess_data.py!",
          "\nThe execution time of preprocess_data.py script is ", execution_time)
