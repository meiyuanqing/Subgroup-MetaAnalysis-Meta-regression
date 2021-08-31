#!/usr/bin/env python
# encoding:utf-8
"""
Author : Yuanqing Mei
Date : 2021/8/31
Time: 15:45
File: separate_tau_subgroup_random_meta_analysis.py
HomePage : http://github.com/yuanqingmei
Email : dg1533019@smail.nju.edu.cn

According to M.Borenstein et. al,[1] (p164), perform subgroup meta-analysis using random-effects weights,
   with a separate estimate of Tau-square for each subgroup.

References:
[1] M.Borenstein, L.V. Hedges, J.P.T. Higgins, H.R. Rothstein. Introduction to meta-analysis, John Wiley & Sons, 2009;
"""
import time


# input: two anonymous arrays, namely effect_size stores each study's effect size and its variance
# output: the results of random effects model, including
#         (1) randomMean：the average of effect sizes;
#         (2) randomStdError: the standard error corresponding to the average of effect sizes.
def random_effect_meta_analysis(effect_size, variance):
    from scipy.stats import norm  # norm.cdf() the cumulative normal distribution function in Python
    # Calculation of p-values based on the chi-square distribution: p_value=1.0-stats.chi2.cdf(chisquare,freedom_degree)
    from scipy import stats
    import numpy as np

    sum_Wi = 0
    sum_WiWi = 0
    sum_WiYi = 0  # Sum(Wi*Yi), where i ranges from 1 to k, and k is the number of studies
    sum_WiYiYi = 0  # Sum(Wi*Yi*Yi), where i ranges from 1 to k, and k is the number of studies

    sum_Wistar = 0
    sum_WistarYi = 0
    d = {}  # return a dict

    study_number = len(variance)
    fixed_weight = [0 for i in range(study_number)]  # 固定模型对应的权值
    random_weight = [0 for i in range(study_number)]  # 随机模型对应的权值

    for i in range(study_number):
        if variance[i] == 0:
            continue
        fixed_weight[i] = 1 / variance[i]
        sum_Wi = sum_Wi + fixed_weight[i]
        sum_WiWi = sum_WiWi + fixed_weight[i] * fixed_weight[i]
        sum_WiYi = sum_WiYi + effect_size[i] * fixed_weight[i]
        sum_WiYiYi = sum_WiYiYi + fixed_weight[i] * effect_size[i] * effect_size[i]

    Q = sum_WiYiYi - sum_WiYi * sum_WiYi / sum_Wi
    df = study_number - 1
    C = sum_Wi - sum_WiWi / sum_Wi

    # When there is only one study in the meta-analysis process, there is no between-study variance, so it set to 0.
    if study_number == 1:
        T2 = 0
    else:
        T2 = (Q - df) / C  # sample estimate of tau square

    if T2 < 0:
        T2 = 0  # 20210411，Set to 0 if T2 is less than 0.   M.Borenstein[2009] P114

    for i in range(study_number):
        random_weight[i] = 1 / (variance[i] + T2)

    for i in range(study_number):
        sum_Wistar = sum_Wistar + random_weight[i]
        sum_WistarYi = sum_WistarYi + random_weight[i] * effect_size[i]

    randomMean = sum_WistarYi / sum_Wistar  # 随机模型元分析后得到的效应平均值
    randomStdError = (1 / sum_Wistar) ** 0.5  # 随机模型元分析的效应平均值对应的标准错
    # When there is only one study in the meta-analysis process, there is no heterogeneity, so it set to 0.
    if study_number == 1:
        I2 = 0
    else:
        I2 = ((Q - df) / Q) * 100  # Higgins et al. (2003) proposed using a statistic, I2,
        # the proportion of the observed variance reflects real differences in effect size
    if I2 < 0:
        I2 = 0  # 20210418，Set to 0 if I2 is less than 0.   M.Borenstein[2009] P110

    pValue_Q = 1.0 - stats.chi2.cdf(Q, df)  # pValue_Q = 1.0 - stats.chi2.cdf(chisquare, freedom_degree)

    d["C"] = C
    d["mean"] = randomMean
    d["stdError"] = randomStdError
    d["LL_CI"] = randomMean - 1.96 * randomStdError  # The 95% lower limits for the summary effect
    d["UL_CI"] = randomMean + 1.96 * randomStdError  # The 95% upper limits for the summary effect
    # 20210719 adds the 84% CI for the summary effect
    d["LL_CI_84"] = randomMean - 1.4051 * randomStdError  # The 84% lower limits for the summary effect
    d["UL_CI_84"] = randomMean + 1.4051 * randomStdError  # The 84% upper limits for the summary effect

    d["ZValue"] = randomMean / randomStdError  # a Z-value to test the null hypothesis that the mean effect is zero
    d["pValue_Z"] = 2 * (1 - norm.cdf(np.abs(randomMean / randomStdError)))  # norm.cdf() 返回标准正态累积分布函数值
    # 20210414 双侧检验时需要增加绝对值符号np.abs
    d["Q"] = Q
    d["df"] = df
    d["pValue_Q"] = pValue_Q
    d["I2"] = I2
    d["tau"] = T2 ** 0.5
    d["LL_ndPred"] = randomMean - 1.96 * (T2 ** 0.5)  # tau、randomMean 已知情况下的新出现的study的effctsize所落的区间
    d["UL_ndPred"] = randomMean + 1.96 * (T2 ** 0.5)  # tau、randomMean 已知情况下的新出现的study的effctsize所落的区间
    # tau、randomMean 未知情况（估计）下的新出现的study的effctsize所落的区间
    # stats.t.ppf(0.975,df)返回学生t分布单尾alpha=0.025区间点(双尾是alpha=0.05)的函数，它是stats.t.cdf()累积分布函数的逆函数
    d["LL_tdPred"] = randomMean - stats.t.ppf(0.975, df) * ((T2 + randomStdError * randomStdError) ** 0.5)
    # tau、randomMean 未知情况（估计）下的新出现的study的effctsize所落的区间
    d["UL_tdPred"] = randomMean + stats.t.ppf(0.975, df) * ((T2 + randomStdError * randomStdError) ** 0.5)

    fixedMean = sum_WiYi / sum_Wi  # 固定模型元分析后得到的效应平均值
    fixedStdError = (1 / sum_Wi) ** 0.5  # 固定模型元分析的效应平均值对应的标准错
    d['fixedMean'] = fixedMean
    d['fixedStdError'] = fixedStdError
    return d


# input: three anonymous arrays, namely effect_size stores each study's effect size, its variance and it subgroup mark.
# output: the results of subgroup meta-analysis with random effects model, including
#         (1) randomMean of all subgroup：the average of effect sizes;
#         (2) randomStdError of all subgroup: the standard error corresponding to the average of effect sizes;
#         (3) other summary statistics in page 167 of M.Borenstein et. al [1];
#         (4) the results of Q -test based on analysis of variance;
#         (5) the results of Q -test for heterogeneity.
def separate_tau_subgroup_random_effect_meta_analysis(effect_size, variance, subgroup):
    import os
    import csv
    from scipy.stats import norm  # norm.cdf() the cumulative normal distribution function in Python
    # Calculation of p-values based on the chi-square distribution: p_value=1.0-stats.chi2.cdf(chisquare,freedom_degree)
    from scipy import stats
    import numpy as np
    import pandas as pd


if __name__ == '__main__':
    s_time = time.time()
    separate_tau_subgroup_random_effect_meta_analysis()
    e_time = time.time()
    execution_time = e_time - s_time

    print("The __name__ is ", __name__, ". This is end of separate_tau_subgroup_random_meta_analysis.py! ",
          "And the elapsed time is ", execution_time, " s.")
