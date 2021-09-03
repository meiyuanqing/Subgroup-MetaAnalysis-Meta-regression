#!/usr/bin/env python
# encoding:utf-8
"""
Author : Yuanqing Mei
Date : 2021/8/31
Time: 15:45
File: subgroup_random_meta_analysis.py
HomePage : http://github.com/yuanqingmei
Email : dg1533019@smail.nju.edu.cn

According to M.Borenstein et. al,[1] (p164), perform subgroup meta-analysis using random-effects weights,
   with a separate estimate of Tau-squared for each subgroup.

References:
[1] M.Borenstein, L.V. Hedges, J.P.T. Higgins, H.R. Rothstein. Introduction to meta-analysis, John Wiley & Sons, 2009;
"""


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
    fixed_weight = [0 for i in range(study_number)]  # Initialization List for weight of fixed effect
    random_weight = [0 for i in range(study_number)]  # Initialization List for weight of random effects

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
        T2 = (Q - df) / C  # sample estimate of tau squared

    if T2 < 0:
        T2 = 0  # 20210411，Set to 0 if T2 is less than 0.   M.Borenstein[2009] P114

    for i in range(study_number):
        random_weight[i] = 1 / (variance[i] + T2)

    for i in range(study_number):
        sum_Wistar = sum_Wistar + random_weight[i]
        sum_WistarYi = sum_WistarYi + random_weight[i] * effect_size[i]

    randomMean = sum_WistarYi / sum_Wistar  # average of random effects
    randomStdError = (1 / sum_Wistar) ** 0.5  # standard error average of random effects
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
#         (4) the results of Q -test based on analysis of variance, which is identical to Q -test for heterogeneity;
#         (5) quantify the magnitude of the difference.(page 160 of M.Borenstein et. al [1])
def subgroup_random_effect_meta_analysis(effect_size, effect_variance, effect_subgroup):

    from scipy.stats import norm  # norm.cdf() the cumulative normal distribution function in Python
    # Calculation of p-values based on the chi-square distribution: p_value=1.0-stats.chi2.cdf(chisquare,freedom_degree)
    from scipy import stats
    import numpy as np
    import pandas as pd

    d = {}  # return a dict of meta-analysis results, including each subgroup, separate, and pooled estimate tau-squared
    subgroups = sorted(set(effect_subgroup))
    # subgroups = sorted(set(subgroup.tolist()))
    print("the subgroups are ", subgroups)
    dic = {"effect_size": effect_size, "effect_size_variance": effect_variance, "effect_size_subgroup": effect_subgroup}

    df_effect_size = pd.DataFrame(dic)

    # pooled all subgroups
    pooled_sum_Wi = 0
    pooled_sum_WiWi = 0
    pooled_sum_WiYi = 0  # Sum(Wi*Yi), where i ranges from 1 to k, and k is the number of studies of all subgroups
    pooled_sum_WiYiYi = 0  # Sum(Wi*Yi*Yi), where i ranges from 1 to k, and k is the number of studies of all subgroups

    pooled_Q = 0
    pooled_df = 0
    pooled_C = 0

    pooled_sum_Wistar = 0
    pooled_sum_WistarYi = 0

    separate_sum_Wistar = 0
    separate_sum_WistarYi = 0

    for subgroup in subgroups:

        sum_Wi = 0
        sum_WiWi = 0
        sum_WiYi = 0  # Sum(Wi*Yi), where i ranges from 1 to k, and k is the number of studies
        sum_WiYiYi = 0  # Sum(Wi*Yi*Yi), where i ranges from 1 to k, and k is the number of studies
        sum_Wistar = 0
        sum_WistarYi = 0

        effect_size = df_effect_size[df_effect_size["effect_size_subgroup"] == subgroup].loc[:,
                      "effect_size"].values.tolist()
        variance = df_effect_size[df_effect_size["effect_size_subgroup"] == subgroup].loc[:,
                   "effect_size_variance"].values.tolist()
        study_number = len(variance)
        fixed_weight = [0 for i in range(study_number)]
        random_weight = [0 for i in range(study_number)]
        print("the current subgroup is ", subgroup, ". the study_number is ", study_number)

        for i in range(study_number):
            if variance[i] == 0:
                continue
            fixed_weight[i] = 1 / variance[i]
            sum_Wi = sum_Wi + fixed_weight[i]
            sum_WiWi = sum_WiWi + fixed_weight[i] * fixed_weight[i]
            sum_WiYi = sum_WiYi + effect_size[i] * fixed_weight[i]
            sum_WiYiYi = sum_WiYiYi + fixed_weight[i] * effect_size[i] * effect_size[i]
            # prepared for summary effects of fixed-effect model within subgroups
            pooled_sum_Wi = pooled_sum_Wi + fixed_weight[i]
            pooled_sum_WiWi = pooled_sum_WiWi + fixed_weight[i] * fixed_weight[i]
            pooled_sum_WiYi = pooled_sum_WiYi + effect_size[i] * fixed_weight[i]
            pooled_sum_WiYiYi = pooled_sum_WiYiYi + fixed_weight[i] * effect_size[i] * effect_size[i]

        Q = sum_WiYiYi - sum_WiYi * sum_WiYi / sum_Wi
        df = study_number - 1
        C = sum_Wi - sum_WiWi / sum_Wi

        print("The sum_WiYiYi value is ", sum_WiYiYi, ". The sum_WiYi value is ", sum_WiYi, ". The sum_Wi value is ",
              sum_Wi, ". The sum_WiWi value is ", sum_WiWi, ". The Q value is ", Q, ". The C value is ", C)

        pooled_Q = pooled_Q + sum_WiYiYi - sum_WiYi * sum_WiYi / sum_Wi
        pooled_df = pooled_df + study_number - 1
        pooled_C = pooled_C + sum_Wi - sum_WiWi / sum_Wi

        # When there is only one study in the meta-analysis, there is no between-study variance, so it set to 0.
        if study_number == 1:
            T2 = 0
        else:
            T2 = (Q - df) / C  # sample estimate of tau squared

        if T2 < 0:
            T2 = 0  # 20210411，Set to 0 if T2 is less than 0.   M.Borenstein[2009] P114

        for i_s in range(study_number):
            random_weight[i_s] = 1 / (variance[i_s] + T2)

        for i_s in range(study_number):
            sum_Wistar = sum_Wistar + random_weight[i_s]
            sum_WistarYi = sum_WistarYi + random_weight[i_s] * effect_size[i_s]

            separate_sum_Wistar = separate_sum_Wistar + random_weight[i_s]
            separate_sum_WistarYi = separate_sum_WistarYi + random_weight[i_s] * effect_size[i_s]

        randomMean = sum_WistarYi / sum_Wistar  # average effect size of each subgroup for separate estimate tau
        randomStdError = (1 / sum_Wistar) ** 0.5  # standard error for average effect size of each subgroup
        # When there is only one study in the meta-analysis process, there is no heterogeneity, so it set to 0.
        if study_number == 1:
            I2 = 0
        else:
            I2 = ((Q - df) / Q) * 100  # Higgins et al. (2003) proposed using a statistic, I2,
            # the proportion of the observed variance reflects real differences in effect size
        if I2 < 0:
            I2 = 0  # 20210418，Set to 0 if I2 is less than 0.   M.Borenstein[2009] P110

        pValue_Q = 1.0 - stats.chi2.cdf(Q, df)  # pValue_Q = 1.0 - stats.chi2.cdf(chisquare, freedom_degree)
        # variance = df_effect_size[df_effect_size["effect_size_subgroup"] == subgroup].loc[:,
        #            "effect_size_variance"].values.tolist()
        # study_number = len(variance)

        d["separate_" + subgroup + "_C"] = C
        d["separate_" + subgroup + "_mean"] = randomMean
        d["separate_" + subgroup + "_stdError"] = randomStdError
        d["separate_" + subgroup + "_LL_CI"] = randomMean - 1.96 * randomStdError  # The 95% lower limits
        d["separate_" + subgroup + "_UL_CI"] = randomMean + 1.96 * randomStdError  # The 95% upper limits
        # 20210719 adds the 84% CI for the summary effect
        d["separate_" + subgroup + "_LL_CI_84"] = randomMean - 1.4051 * randomStdError  # The 84% lower limits
        d["separate_" + subgroup + "_UL_CI_84"] = randomMean + 1.4051 * randomStdError  # The 84% upper limits
        # a Z-value to test the null hypothesis that the mean effect is zero;norm.cdf() 返回标准正态累积分布函数值
        d["separate_" + subgroup + "_ZValue"] = randomMean / randomStdError
        d["separate_" + subgroup + "_pValue_Z"] = 2 * (1 - norm.cdf(np.abs(randomMean / randomStdError)))
        # 20210414 双侧检验时需要增加绝对值符号np.abs
        d["separate_" + subgroup + "_Q"] = Q
        d["separate_" + subgroup + "_df"] = df
        d["separate_" + subgroup + "_pValue_Q"] = pValue_Q
        d["separate_" + subgroup + "_I2"] = I2
        d["separate_" + subgroup + "_tau"] = T2 ** 0.5
        # tau、randomMean 已知情况下的新出现的study的effctsize所落的区间
        d["separate_" + subgroup + "_LL_ndPred"] = randomMean - 1.96 * (T2 ** 0.5)
        d["separate_" + subgroup + "_UL_ndPred"] = randomMean + 1.96 * (T2 ** 0.5)
        # tau、randomMean 未知情况（估计）下的新出现的study的effctsize所落的区间
        # stats.t.ppf(0.975,df)返回学生t分布单尾alpha=0.025区间点(双尾是alpha=0.05)的函数，它是stats.t.cdf()累积分布函数的逆函数
        d["separate_" + subgroup + "_LL_tdPred"] = randomMean - stats.t.ppf(0.975, df) * (
                (T2 + randomStdError * randomStdError) ** 0.5)
        # tau、randomMean 未知情况（估计）下的新出现的study的effctsize所落的区间
        d["separate_" + subgroup + "_UL_tdPred"] = randomMean + stats.t.ppf(0.975, df) * (
                (T2 + randomStdError * randomStdError) ** 0.5)

        fixedMean = sum_WiYi / sum_Wi  # 固定模型元分析后得到的效应平均值
        fixedStdError = (1 / sum_Wi) ** 0.5  # 固定模型元分析的效应平均值对应的标准错
        d[subgroup + '_fixedMean'] = fixedMean
        d[subgroup + '_fixedStdError'] = fixedStdError

    # compute the pooled estimate tau of summary effect from all subgroups
    tau_squared_within = (pooled_Q - pooled_df) / pooled_C  # sample estimate of tau squared
    if tau_squared_within < 0:
        tau_squared_within = 0  # 20210411，Set to 0 if T2 is less than 0.   M.Borenstein[2009] P114

    for subgroup_pooled in subgroups:

        print("the current subgroup is ", subgroup_pooled)
        sum_Wi = 0
        sum_WiWi = 0
        sum_WiYi = 0  # Sum(Wi*Yi), where i ranges from 1 to k, and k is the number of studies
        sum_WiYiYi = 0  # Sum(Wi*Yi*Yi), where i ranges from 1 to k, and k is the number of studies

        sum_Wistar = 0
        sum_WistarYi = 0

        effect_size = df_effect_size[df_effect_size["effect_size_subgroup"] == subgroup_pooled].loc[:,
                      "effect_size"].values.tolist()
        variance = df_effect_size[df_effect_size["effect_size_subgroup"] == subgroup_pooled].loc[:,
                   "effect_size_variance"].values.tolist()
        study_number = len(variance)

        print("the study_number is ", study_number)
        print("the effect_size is ", effect_size)
        print("the variance is ", variance)

        fixed_weight = [0 for i in range(study_number)]
        random_weight = [0 for i in range(study_number)]

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

        print("The Q value is ", Q, ". The C value is ", C)

        for i_s in range(study_number):
            random_weight[i_s] = 1 / (variance[i_s] + tau_squared_within)

        for i_s in range(study_number):
            sum_Wistar = sum_Wistar + random_weight[i_s]
            sum_WistarYi = sum_WistarYi + random_weight[i_s] * effect_size[i_s]

            pooled_sum_Wistar = pooled_sum_Wistar + random_weight[i_s]
            pooled_sum_WistarYi = pooled_sum_WistarYi + random_weight[i_s] * effect_size[i_s]

        randomMean = sum_WistarYi / sum_Wistar  # average effect size of each subgroup for separate estimate tau
        randomStdError = (1 / sum_Wistar) ** 0.5  # standard error for average effect size of each subgroup
        # When there is only one study in the meta-analysis process, there is no heterogeneity, so it set to 0.
        # the proportion of the observed variance reflects real differences in effect size
        if study_number == 1:
            I2 = 0
        elif ((Q - df) / Q) < 0:
            I2 = 0  # 20210418，Set to 0 if I2 is less than 0.   M.Borenstein[2009] P110
        else:
            I2 = ((Q - df) / Q) * 100  # Higgins et al. (2003) proposed using a statistic, I2,

        pValue_Q = 1.0 - stats.chi2.cdf(Q, df)

        d["pooled_" + subgroup_pooled + "_C"] = C
        d["pooled_" + subgroup_pooled + "_mean"] = randomMean
        d["pooled_" + subgroup_pooled + "_stdError"] = randomStdError
        d["pooled_" + subgroup_pooled + "_LL_CI"] = randomMean - 1.96 * randomStdError  # The 95% lower limits
        d["pooled_" + subgroup_pooled + "_UL_CI"] = randomMean + 1.96 * randomStdError  # The 95% upper limits
        # 20210719 adds the 84% CI for the summary effect
        d["pooled_" + subgroup_pooled + "_LL_CI_84"] = randomMean - 1.4051 * randomStdError  # The 84% lower limits
        d["pooled_" + subgroup_pooled + "_UL_CI_84"] = randomMean + 1.4051 * randomStdError  # The 84% upper limits
        # a Z-value to test the null hypothesis that the mean effect is zero;norm.cdf() 返回标准正态累积分布函数值
        d["pooled_" + subgroup_pooled + "_ZValue"] = randomMean / randomStdError
        d["pooled_" + subgroup_pooled + "_pValue_Z"] = 2 * (1 - norm.cdf(np.abs(randomMean / randomStdError)))
        # 20210414 双侧检验时需要增加绝对值符号np.abs
        d["pooled_" + subgroup_pooled + "_Q"] = Q
        d["pooled_" + subgroup_pooled + "_df"] = df
        d["pooled_" + subgroup_pooled + "_pValue_Q"] = pValue_Q
        d["pooled_" + subgroup_pooled + "_I2"] = I2
        d["pooled_" + subgroup_pooled + "_tau"] = T2 ** 0.5
        # tau、randomMean 已知情况下的新出现的study的effctsize所落的区间
        d["pooled_" + subgroup_pooled + "_LL_ndPred"] = randomMean - 1.96 * (T2 ** 0.5)
        d["pooled_" + subgroup_pooled + "_UL_ndPred"] = randomMean + 1.96 * (T2 ** 0.5)
        # tau、randomMean 未知情况（估计）下的新出现的study的effctsize所落的区间
        # stats.t.ppf(0.975,df)返回学生t分布单尾alpha=0.025区间点(双尾是alpha=0.05)的函数，它是stats.t.cdf()累积分布函数的逆函数
        d["pooled_" + subgroup_pooled + "_LL_tdPred"] = randomMean - stats.t.ppf(0.975, df) * (
                (T2 + randomStdError * randomStdError) ** 0.5)
        # tau、randomMean 未知情况（估计）下的新出现的study的effctsize所落的区间
        d["pooled_" + subgroup_pooled + "_UL_tdPred"] = randomMean + stats.t.ppf(0.975, df) * (
                (T2 + randomStdError * randomStdError) ** 0.5)

    pooled_randomMean = pooled_sum_WistarYi / pooled_sum_Wistar  # 随机模型元分析后得到的效应平均值
    pooled_randomStdError = (1 / pooled_sum_Wistar) ** 0.5  # 随机模型元分析的效应平均值对应的标准错

    # When there is only one study in the meta-analysis, there is no between-study variance, so it set to 0.
    if study_number == 1:
        pooled_T2 = 0
    elif(pooled_Q - pooled_df) / pooled_C < 0:     # 20210411，Set to 0 if T2 is less than 0.   M.Borenstein[2009] P114
        pooled_T2 = 0
    else:
        pooled_T2 = (pooled_Q - pooled_df) / pooled_C

    # When there is only one study in the meta-analysis process, there is no heterogeneity, so it set to 0.
    # the proportion of the observed variance reflects real differences in effect size
    if study_number == 1:
        pooled_I2 = 0
    elif ((pooled_Q - pooled_df) / pooled_Q) < 0: # 20210418，Set to 0 if I2 is less than 0.   M.Borenstein[2009] P110
        pooled_I2 = 0
    else:
        pooled_I2 = ((pooled_Q - pooled_df) / pooled_Q) * 100  # Higgins et al. (2003) proposed using a statistic, I2,

    pooled_pValue_Q = 1.0 - stats.chi2.cdf(pooled_Q, pooled_df)

    d["pooled_C"] = pooled_C
    d["pooled_mean"] = pooled_randomMean
    d["pooled_stdError"] = pooled_randomStdError
    d["pooled_LL_CI"] = pooled_randomMean - 1.96 * pooled_randomStdError  # The 95% lower limits
    d["pooled_L_CI"] = pooled_randomMean + 1.96 * pooled_randomStdError  # The 95% upper limits
    # 20210719 adds the 84% CI for the summary effect
    d["pooled_LL_CI_84"] = pooled_randomMean - 1.4051 * pooled_randomStdError  # The 84% lower limits
    d["pooled_UL_CI_84"] = pooled_randomMean + 1.4051 * pooled_randomStdError  # The 84% upper limits
    # a Z-value to test the null hypothesis that the mean effect is zero
    d["pooled_ZValue"] = pooled_randomMean / pooled_randomStdError
    # norm.cdf() 返回标准正态累积分布函数值, 20210414 双侧检验时需要增加绝对值符号np.abs
    d["pooled_pValue_Z"] = 2 * (1 - norm.cdf(np.abs(pooled_randomMean / pooled_randomStdError)))
    d["pooled_Q"] = pooled_Q
    d["pooled_df"] = pooled_df
    d["pooled_pValue_Q"] = pooled_pValue_Q
    d["pooled_I2"] = pooled_I2
    d["pooled_tau"] = pooled_T2 ** 0.5
    # tau、randomMean 已知情况下的新出现的study的effctsize所落的区间
    d["pooled_LL_ndPred"] = pooled_randomMean - 1.96 * (pooled_T2 ** 0.5)
    d["pooled_UL_ndPred"] = pooled_randomMean + 1.96 * (pooled_T2 ** 0.5)
    # tau、randomMean 未知情况（估计）下的新出现的study的effctsize所落的区间
    # stats.t.ppf(0.975,df)返回学生t分布单尾alpha=0.025区间点(双尾是alpha=0.05)的函数，它是stats.t.cdf()累积分布函数的逆函数
    d["pooled_LL_tdPred"] = pooled_randomMean - stats.t.ppf(0.975, pooled_df) \
                            * ((pooled_T2 + pooled_randomStdError * pooled_randomStdError) ** 0.5)
    # tau、randomMean 未知情况（估计）下的新出现的study的effctsize所落的区间
    d["pooled_UL_tdPred"] = pooled_randomMean + stats.t.ppf(0.975, pooled_df) \
                            * ((pooled_T2 + pooled_randomStdError * pooled_randomStdError) ** 0.5)

    pooled_fixedMean = pooled_sum_WiYi / pooled_sum_Wi  # 固定模型元分析后得到的效应平均值
    pooled_fixedStdError = (1 / pooled_sum_Wi) ** 0.5  # 固定模型元分析的效应平均值对应的标准错
    d['pooled_fixedMean'] = pooled_fixedMean
    d['pooled_fixedStdError'] = pooled_fixedStdError

    # a Q-test based on analysis of variance, which is identical to Z test and Q-test based on heterogeneity
    Q_total = pooled_sum_WiYiYi - pooled_sum_WiYi * pooled_sum_WiYi / pooled_sum_Wi
    Q_within = pooled_Q
    Q_between = Q_total - Q_within
    df_between = len(subgroups) - 1
    pValue_Q_between = 1.0 - stats.chi2.cdf(Q_between, df_between)

    d["Q-test_ANOVA"] = pValue_Q_between
    d["Q-test_Q_total"] = Q_total
    d["Q-test_Q_within"] = Q_within
    d["Q-test_Q_between"] = Q_between

    # Quantify the magnitude of the difference

    # Compute the separate estimate tau-squared：M.Borenstein[2009] P179 did not report the statistic
    separate_T2 = (pooled_Q - pooled_df) / pooled_C  # sample estimate of tau squared

    if separate_T2 < 0:
        separate_T2 = 0  # 20210411，Set to 0 if T2 is less than 0.   M.Borenstein[2009] P114
    separate_randomMean = separate_sum_WistarYi / separate_sum_Wistar  # 随机模型元分析后得到的效应平均值
    separate_randomStdError = (1 / separate_sum_Wistar) ** 0.5  # 随机模型元分析的效应平均值对应的标准错
    separate_I2 = ((pooled_Q - pooled_df) / pooled_Q) * 100  # Higgins et al. (2003) proposed using a statistic, I2,
    # the proportion of the observed variance reflects real differences in effect size
    if separate_I2 < 0:
        separate_I2 = 0  # 20210418，Set to 0 if I2 is less than 0.   M.Borenstein[2009] P110

    # pValue_Q = 1.0 - stats.chi2.cdf(chisquare, freedom_degree)
    separate_pValue_Q = 1.0 - stats.chi2.cdf(pooled_Q, pooled_df)

    d["separate_C"] = pooled_C
    d["separate_mean"] = separate_randomMean
    d["separate_stdError"] = separate_randomStdError
    d["separate_LL_CI"] = separate_randomMean - 1.96 * separate_randomStdError  # The 95% lower limits
    d["separate_L_CI"] = separate_randomMean + 1.96 * separate_randomStdError  # The 95% upper limits
    # 20210719 adds the 84% CI for the summary effect
    d["separate_LL_CI_84"] = separate_randomMean - 1.4051 * separate_randomStdError  # The 84% lower limits
    d["separate_UL_CI_84"] = separate_randomMean + 1.4051 * separate_randomStdError  # The 84% upper limits
    # a Z-value to test the null hypothesis that the mean effect is zero
    d["separate_ZValue"] = separate_randomMean / separate_randomStdError
    # norm.cdf() 返回标准正态累积分布函数值, 20210414 双侧检验时需要增加绝对值符号np.abs
    d["separate_pValue_Z"] = 2 * (1 - norm.cdf(np.abs(separate_randomMean / separate_randomStdError)))

    d["separate_Q"] = pooled_Q
    d["separate_df"] = pooled_df
    d["separate_pValue_Q"] = separate_pValue_Q
    d["separate_I2"] = separate_I2
    d["separate_tau"] = separate_T2 ** 0.5
    # tau、randomMean 已知情况下的新出现的study的effctsize所落的区间
    d["separate_LL_ndPred"] = separate_randomMean - 1.96 * (separate_T2 ** 0.5)
    d["separate_UL_ndPred"] = separate_randomMean + 1.96 * (separate_T2 ** 0.5)
    # tau、randomMean 未知情况（估计）下的新出现的study的effctsize所落的区间
    # stats.t.ppf(0.975,df)返回学生t分布单尾alpha=0.025区间点(双尾是alpha=0.05)的函数，它是stats.t.cdf()累积分布函数的逆函数
    d["separate_LL_tdPred"] = separate_randomMean - stats.t.ppf(0.975, pooled_df) \
                              * ((separate_T2 + separate_randomStdError * separate_randomStdError) ** 0.5)
    # tau、randomMean 未知情况（估计）下的新出现的study的effctsize所落的区间
    d["separate_UL_tdPred"] = separate_randomMean + stats.t.ppf(0.975, pooled_df) \
                              * ((separate_T2 + separate_randomStdError * separate_randomStdError) ** 0.5)

    separate_fixedMean = pooled_sum_WiYi / pooled_sum_Wi  # 固定模型元分析后得到的效应平均值
    separate_fixedStdError = (1 / pooled_sum_Wi) ** 0.5  # 固定模型元分析的效应平均值对应的标准错
    d['separate_fixedMean'] = separate_fixedMean
    d['separate_fixedStdError'] = separate_fixedStdError

    return d


if __name__ == '__main__':
    import os
    import sys
    import time
    import pandas as pd

    working_directory = "F:\\NJU\\subMeta\\experiments\\subgroupPearson\\"
    s_time = time.time()
    df = pd.read_csv(working_directory + "LOCsubgroup.csv", keep_default_na=False, na_values=[""])
    df = df.dropna(axis=0, how='any', inplace=False).reset_index(drop=True)
    FisherZ_effect_size = df[df["metric"] == "LOC"].loc[:, "Fisher_Z"].astype(float)
    FisherZ_variance = df[df["metric"] == "LOC"].loc[:, "Fisher_Z_variance"].astype(float)
    FisherZ_subgroup = df[df["metric"] == "LOC"].loc[:, "subgroup"]
    # subgroup_random_effect_meta_analysis(FisherZ_effect_size, FisherZ_variance, FisherZ_subgroup)

    # P173 Table 19.10
    effect_size_A = [0.11, 0.224, 0.338, 0.451, 0.480, 0.440, 0.492, 0.651, 0.710, 0.740]
    variance_A = [0.01, 0.03, 0.02, 0.015, 0.01, 0.015, 0.02, 0.015, 0.025, 0.012]
    subgroup = ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"]
    subgroup_results = subgroup_random_effect_meta_analysis(effect_size_A, variance_A, subgroup)
    print("the subgroup_results is ", subgroup_results)
    for s in subgroup_results:
        print(s, subgroup_results[s])
    e_time = time.time()
    execution_time = e_time - s_time

    print("The __name__ is ", __name__, ". This", os.path.basename(sys.argv[0]), "ended within", execution_time, "(s).")
