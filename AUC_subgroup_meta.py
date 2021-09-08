#!/usr/bin/env python
# encoding:utf-8
"""
Author : Yuanqing Mei
Date : 2021/9/7
Time: 8:46
File: AUC_subgroup_meta.py
HomePage : http://github.com/yuanqingmei
Email : dg1533019@smail.nju.edu.cn

Inputs: the Pearson (transformed by Z_fisher) and AUC of each metric for each subgroup .

Outputs:
(1) the first csv file：the results of subgroup random effects meta-analysis: for each subgroup and all subgroups;
(2) the second csv file：the Z-test results for each and all subgroups;
(3) the third csv file：the Q-test results for each and all subgroups;
(4) the fourth csv file：the R-squared results for each software metric;

References:
[1] M.Borenstein, L.V. Hedges, J.P.T. Higgins, H.R. Rothstein. Introduction to meta-analysis, John Wiley & Sons, 2009;
[2] Comprehensive meta-analysis. http://www.meta-analysis.com.
[3] Feng Zhang,Audris Mockus,Iman Keivanloo,Ying Zou  Towards building a universal defect prediction model with rank
    transformed predictors Springer Science+Business Media , New York 2015.

"""


# Inputs: three anonymous arrays, namely effect_size stores each study's effect size, its variance and it subgroup mark.
# Outputs: the results of subgroup meta-analysis with random effects model, including
#         (1) randomMean of all subgroup：the average of effect sizes for separated and pooled tau, respectively;
#         (2) randomStdError of all subgroup: the standard error corresponding to the average of effect sizes;
#         (3) other summary statistics in page 167 of M.Borenstein et. al [1];
#         (4) the results of Q -test based on analysis of variance, which is identical to Q -test for heterogeneity;
#         (5) Z-test of two subgroups, quantify the magnitude of the difference.(page 160 of M.Borenstein et. al [1])
def subgroup_random_effect_meta_analysis(effect_size, effect_variance, effect_subgroup):
    from scipy.stats import norm  # norm.cdf() the cumulative normal distribution function in Python
    from scipy import stats  # chi-square distribution: p_value=1.0-stats.chi2.cdf(chi-square, freedom_degree)
    import numpy as np
    import pandas as pd

    d = {}  # return a dict of results, including each subgroup, separated, and pooled estimate tau-squared
    subgroups = sorted(set(effect_subgroup))
    print("the subgroups are ", subgroups)
    dic = {"effect_size": effect_size, "effect_size_variance": effect_variance, "effect_size_subgroup": effect_subgroup}

    df_effect_size = pd.DataFrame(dic)

    pooled_sum_Wi = 0  # pooled all subgroups for fixed effect model
    pooled_sum_WiWi = 0
    pooled_sum_WiYi = 0  # Sum(Wi*Yi), where i ranges from 1 to k, and k is the number of studies of all subgroups
    pooled_sum_WiYiYi = 0  # Sum(Wi*Yi*Yi), where i ranges from 1 to k, and k is the number of studies of all subgroups

    pooled_Q = 0  # tau-squared-within is computed using fixed effect's Q, df, and c.
    pooled_df = 0
    pooled_C = 0

    pooled_Q_separate = 0  # used for Q-test for separated estimate tau

    pooled_Q_pooled = 0  # used for Q-test for pooled estimate tau

    separate_sum_Wistar = 0  # combined all subgroups for separated estimate tau
    separate_sum_WistarWistar = 0
    separate_sum_WistarYi = 0
    separate_sum_WistarYiYi = 0

    pooled_sum_Wistar = 0  # combined all subgroups for pooled estimate tau
    pooled_sum_WistarWistar = 0
    pooled_sum_WistarYi = 0
    pooled_sum_WistarYiYi = 0

    for subgroup in subgroups:

        sum_Wi = 0
        sum_WiWi = 0
        sum_WiYi = 0  # Sum(Wi*Yi), where i ranges from 1 to k, and k is the number of studies
        sum_WiYiYi = 0  # Sum(Wi*Yi*Yi), where i ranges from 1 to k, and k is the number of studies

        sum_Wistar = 0
        sum_WistarWistar = 0
        sum_WistarYi = 0
        sum_WistarYiYi = 0

        effect_size_sub = df_effect_size[df_effect_size["effect_size_subgroup"] == subgroup].loc[:,
                          "effect_size"].values.tolist()
        variance = df_effect_size[df_effect_size["effect_size_subgroup"] == subgroup].loc[:,
                   "effect_size_variance"].values.tolist()
        study_number = len(variance)
        fixed_weight = [0 for i in range(study_number)]
        random_weight = [0 for i in range(study_number)]
        print("the current subgroup is ", subgroup, ". the study_number is ", study_number)

        for i in range(study_number):  # prepare for Tau of each subgroup
            if variance[i] == 0:
                continue
            fixed_weight[i] = 1 / variance[i]
            sum_Wi = sum_Wi + fixed_weight[i]
            sum_WiWi = sum_WiWi + fixed_weight[i] * fixed_weight[i]
            sum_WiYi = sum_WiYi + effect_size_sub[i] * fixed_weight[i]
            sum_WiYiYi = sum_WiYiYi + fixed_weight[i] * effect_size_sub[i] * effect_size_sub[i]

            pooled_sum_Wi = pooled_sum_Wi + fixed_weight[i]
            pooled_sum_WiWi = pooled_sum_WiWi + fixed_weight[i] * fixed_weight[i]
            pooled_sum_WiYi = pooled_sum_WiYi + effect_size_sub[i] * fixed_weight[i]
            pooled_sum_WiYiYi = pooled_sum_WiYiYi + fixed_weight[i] * effect_size_sub[i] * effect_size_sub[i]

        Q = sum_WiYiYi - sum_WiYi * sum_WiYi / sum_Wi
        df = study_number - 1
        C = sum_Wi - sum_WiWi / sum_Wi

        if study_number == 1:  # Only one study in meta-analysis results in no between-study variance, so it set to 0.
            T2 = 0
        else:
            T2 = (Q - df) / C  # sample estimate of tau squared

        if T2 < 0:
            T2 = 0  # 20210411，Set to 0 if T2 is less than 0.   M.Borenstein[2009] P114

        for i_s in range(study_number):
            random_weight[i_s] = 1 / (variance[i_s] + T2)

        for i_s in range(study_number):
            sum_Wistar = sum_Wistar + random_weight[i_s]
            sum_WistarWistar = sum_WistarWistar + random_weight[i_s] * random_weight[i_s]
            sum_WistarYi = sum_WistarYi + random_weight[i_s] * effect_size_sub[i_s]
            sum_WistarYiYi = sum_WistarYiYi + random_weight[i_s] * effect_size_sub[i_s] * effect_size_sub[i_s]

            # prepared for summary effects of random-effect model within subgroups
            separate_sum_Wistar = separate_sum_Wistar + random_weight[i_s]
            separate_sum_WistarWistar = separate_sum_WistarWistar + random_weight[i_s] * random_weight[i_s]
            separate_sum_WistarYi = separate_sum_WistarYi + random_weight[i_s] * effect_size_sub[i_s]
            separate_sum_WistarYiYi = separate_sum_WistarYiYi \
                                      + random_weight[i_s] * effect_size_sub[i_s] * effect_size_sub[i_s]

        Q_subgroup = sum_WistarYiYi - sum_WistarYi * sum_WistarYi / sum_Wistar
        df_subgroup = study_number - 1
        C_subgroup = sum_Wistar - sum_WistarWistar / sum_Wistar

        pooled_Q_separate = pooled_Q_separate + Q_subgroup  # Q-test for separate estimate tau subgroup random effect
        # pooled_df_separate = pooled_df + df_subgroup
        # pooled_C_separate = pooled_C + C_subgroup

        pooled_Q = pooled_Q + Q  # tau-squared-within is computed using fixed effect's Q, df, and c.
        pooled_df = pooled_df + df
        pooled_C = pooled_C + C

        randomMean = sum_WistarYi / sum_Wistar  # average effect size of each subgroup for separate estimate tau
        randomVariance = 1 / sum_Wistar  # variance for average effect size of each subgroup
        randomStdError = (1 / sum_Wistar) ** 0.5  # standard error for average effect size of each subgroup

        if study_number == 1:  # Only one study in the meta-analysis results in no heterogeneity, so it set to 0.
            I2 = 0  # the proportion of the observed variance reflects real differences in effect size
        elif ((Q_subgroup - df_subgroup) / Q_subgroup) < 0:
            I2 = 0  # 20210418，Set to 0 if I2 is less than 0.   M.Borenstein[2009] P110
        else:
            I2 = ((Q_subgroup - df_subgroup) / Q_subgroup) * 100  # Higgins et al. (2003) proposed using a statistic, I2

        pValue_Q = 1.0 - stats.chi2.cdf(Q_subgroup, df_subgroup)

        d["separate_" + subgroup + "_mean"] = randomMean
        d["separate_" + subgroup + "_variance"] = randomVariance
        d["separate_" + subgroup + "_stdError"] = randomStdError
        d["separate_" + subgroup + "_LL_CI"] = randomMean - 1.96 * randomStdError  # The 95% lower limits
        d["separate_" + subgroup + "_UL_CI"] = randomMean + 1.96 * randomStdError  # The 95% upper limits
        d["separate_" + subgroup + "_LL_CI_84"] = randomMean - 1.4051 * randomStdError  # The 84% lower limits, 20210719
        d["separate_" + subgroup + "_UL_CI_84"] = randomMean + 1.4051 * randomStdError  # The 84% upper limits
        # a Z-value to test the null hypothesis that the mean effect is zero; norm.cdf() 返回标准正态累积分布函数值,
        d["separate_" + subgroup + "_ZValue"] = randomMean / randomStdError  # 20210414 双侧检验时需要增加绝对值符号np.abs
        d["separate_" + subgroup + "_pValue_Z"] = 2 * (1 - norm.cdf(np.abs(randomMean / randomStdError)))
        d["separate_" + subgroup + "_Q"] = Q_subgroup
        d["separate_" + subgroup + "_df"] = df_subgroup
        d["separate_" + subgroup + "_C"] = C_subgroup
        d["separate_" + subgroup + "_pValue_Q"] = pValue_Q
        d["separate_" + subgroup + "_I2"] = I2
        # d["separate_" + subgroup + "_tau"] = T2 ** 0.5
        # d["separate_" + subgroup + "_LL_ndPred"] = randomMean - 1.96 * (T2 ** 0.5)  # tau、randomMean 已知情况下的新出现的
        # d["separate_" + subgroup + "_UL_ndPred"] = randomMean + 1.96 * (T2 ** 0.5)  # study的effect size所落的区间
        # tau、randomMean 未知情况（估计）下的新出现的study的effect size所落的区间
        # stats.t.ppf(0.975,df)返回学生t分布单尾alpha=0.025区间点(双尾是alpha=0.05)的函数，它是stats.t.cdf()累积分布函数的逆函数
        d["separate_" + subgroup + "_LL_tdPred"] = randomMean - stats.t.ppf(0.975, df_subgroup) * (
                (T2 + randomStdError * randomStdError) ** 0.5)
        d["separate_" + subgroup + "_UL_tdPred"] = randomMean + stats.t.ppf(0.975, df_subgroup) * (
                (T2 + randomStdError * randomStdError) ** 0.5)

        fixedMean = sum_WiYi / sum_Wi  # average effect sizes for fixed model of each subgroup
        fixedVariance = 1 / sum_Wi  # variance of average effect sizes for fixed model of each subgroup
        fixedStdError = (1 / sum_Wi) ** 0.5  # standard error of average effect sizes for fixed model of each subgroup
        d[subgroup + '_fixedMean'] = fixedMean
        d[subgroup + '_fixedVariance'] = fixedVariance
        d[subgroup + '_fixedStdError'] = fixedStdError

    # TODO: M.Borenstein[2009] P155, the rest of summary statistics in Table 19.2 can be computed here.
    pooled_fixedMean = pooled_sum_WiYi / pooled_sum_Wi  # average effect sizes for fixed model of all subgroups
    pooled_fixedVariance = 1 / pooled_sum_Wi  # variance
    pooled_fixedStdError = (1 / pooled_sum_Wi) ** 0.5  # standard error
    pooled_Q_fixed = pooled_sum_WiYiYi - pooled_sum_WiYi * pooled_sum_WiYi / pooled_sum_Wi
    pooled_C_fixed = pooled_sum_Wi - pooled_sum_WiWi / pooled_sum_Wi
    pooled_df_fixed = len(effect_size) - 1
    combined_T2 = (pooled_Q_fixed - pooled_df_fixed) / pooled_C_fixed
    combined_I2 = ((pooled_Q_fixed - pooled_df_fixed) / pooled_Q_fixed) * 100
    d['combined__fixedMean'] = pooled_fixedMean
    d['combined_fixedVariance'] = pooled_fixedVariance
    d['combined_fixedStdError'] = pooled_fixedStdError
    d['combined_tau_squared'] = combined_T2
    d['combined_I_squared'] = combined_I2

    # compute the pooled estimate tau of summary effect from all subgroups
    tau_squared_within = (pooled_Q - pooled_df) / pooled_C  # sample estimate of tau squared
    if tau_squared_within < 0:
        tau_squared_within = 0  # 20210411，Set to 0 if T2 is less than 0.   M.Borenstein[2009] P114

    # print("the tau_squared_within is ", tau_squared_within, "the pooled_Q_fixed is ", pooled_Q_fixed,
    #       "the pooled_df_fixed is ", pooled_df_fixed, "the pooled_C_fixed is ", pooled_C_fixed)

    # compute the R2 P181
    d['R2'] = 1 - (tau_squared_within / combined_T2)

    for subgroup_pooled in subgroups:

        print("the current subgroup is ", subgroup_pooled)
        sum_Wistar = 0
        sum_WistarWistar = 0
        sum_WistarYi = 0
        sum_WistarYiYi = 0

        effect_size = df_effect_size[df_effect_size["effect_size_subgroup"] == subgroup_pooled].loc[:,
                      "effect_size"].values.tolist()
        variance = df_effect_size[df_effect_size["effect_size_subgroup"] == subgroup_pooled].loc[:,
                   "effect_size_variance"].values.tolist()
        study_number = len(variance)

        print("the study_number is ", study_number, "the effect_size is ", effect_size, "the variance is ", variance)

        random_weight = [0 for i in range(study_number)]

        for i_s in range(study_number):
            random_weight[i_s] = 1 / (variance[i_s] + tau_squared_within)

        for i_s in range(study_number):
            sum_Wistar = sum_Wistar + random_weight[i_s]
            sum_WistarWistar = sum_WistarWistar + random_weight[i_s] * random_weight[i_s]
            sum_WistarYi = sum_WistarYi + random_weight[i_s] * effect_size[i_s]
            sum_WistarYiYi = sum_WistarYiYi + random_weight[i_s] * effect_size[i_s] * effect_size[i_s]

            pooled_sum_Wistar = pooled_sum_Wistar + random_weight[i_s]
            pooled_sum_WistarWistar = pooled_sum_WistarWistar + random_weight[i_s] * random_weight[i_s]
            pooled_sum_WistarYi = pooled_sum_WistarYi + random_weight[i_s] * effect_size[i_s]
            pooled_sum_WistarYiYi = pooled_sum_WistarYiYi + random_weight[i_s] * effect_size[i_s] * effect_size[i_s]

        Q_subgroup = sum_WistarYiYi - sum_WistarYi * sum_WistarYi / sum_Wistar
        df_subgroup = study_number - 1
        C_subgroup = sum_Wistar - sum_WistarWistar / sum_Wistar

        # Q-test for pooled estimate tau subgroup random effect meta-analysis
        pooled_Q_pooled = pooled_Q_pooled + Q_subgroup

        randomMean = sum_WistarYi / sum_Wistar  # average effect size of each subgroup for separate estimate tau
        randomVariance = 1 / sum_Wistar  # variance for average effect size of each subgroup
        randomStdError = (1 / sum_Wistar) ** 0.5  # standard error for average effect size of each subgroup

        if study_number == 1:  # Only one study in the meta-analysis process results in no heterogeneity, so it set to 0
            I2 = 0  # the proportion of the observed variance reflects real differences in effect size
        elif ((Q_subgroup - df_subgroup) / Q_subgroup) < 0:
            I2 = 0  # 20210418，Set to 0 if I2 is less than 0.   M.Borenstein[2009] P110
        else:
            I2 = ((Q_subgroup - df_subgroup) / Q_subgroup) * 100  # Higgins et al. (2003) proposed using a statistic, I2

        pValue_Q = 1.0 - stats.chi2.cdf(Q_subgroup, df_subgroup)

        d["pooled_" + subgroup_pooled + "_mean"] = randomMean
        d["pooled_" + subgroup_pooled + "_variance"] = randomVariance
        d["pooled_" + subgroup_pooled + "_randomStdError"] = randomStdError
        d["pooled_" + subgroup_pooled + "_LL_CI"] = randomMean - 1.96 * randomStdError  # The 95% lower limits
        d["pooled_" + subgroup_pooled + "_UL_CI"] = randomMean + 1.96 * randomStdError  # The 95% upper limits
        d["pooled_" + subgroup_pooled + "_LL_CI_84"] = randomMean - 1.4051 * randomStdError  # The 84% lower limits
        d["pooled_" + subgroup_pooled + "_UL_CI_84"] = randomMean + 1.4051 * randomStdError  # The 84% upper limits
        # a Z-value to test the null hypothesis that the mean effect is zero;norm.cdf() 返回标准正态累积分布函数值
        d["pooled_" + subgroup_pooled + "_ZValue"] = randomMean / randomStdError  # 20210414双侧检验时需要增加绝对值符号np.abs
        d["pooled_" + subgroup_pooled + "_pValue_Z"] = 2 * (1 - norm.cdf(np.abs(randomMean / randomStdError)))
        d["pooled_" + subgroup_pooled + "_Q"] = Q_subgroup
        d["pooled_" + subgroup_pooled + "_df"] = df_subgroup
        d["pooled_" + subgroup_pooled + "_C"] = C_subgroup
        d["pooled_" + subgroup_pooled + "_pValue_Q"] = pValue_Q
        d["pooled_" + subgroup_pooled + "_I2"] = I2
        # d["pooled_" + subgroup_pooled + "_tau"] = T2 ** 0.5
        # # tau、randomMean 已知情况下的新出现的study的effctsize所落的区间
        # d["pooled_" + subgroup_pooled + "_LL_ndPred"] = randomMean - 1.96 * (T2 ** 0.5)
        # d["pooled_" + subgroup_pooled + "_UL_ndPred"] = randomMean + 1.96 * (T2 ** 0.5)
        # tau、randomMean 未知情况（估计）下的新出现的study的effctsize所落的区间
        # stats.t.ppf(0.975,df)返回学生t分布单尾alpha=0.025区间点(双尾是alpha=0.05)的函数，它是stats.t.cdf()累积分布函数的逆函数
        d["pooled_" + subgroup_pooled + "_LL_tdPred"] = randomMean - stats.t.ppf(0.975, df_subgroup) * (
                (T2 + randomStdError * randomStdError) ** 0.5)
        # tau、randomMean 未知情况（估计）下的新出现的study的effctsize所落的区间
        d["pooled_" + subgroup_pooled + "_UL_tdPred"] = randomMean + stats.t.ppf(0.975, df_subgroup) * (
                (T2 + randomStdError * randomStdError) ** 0.5)

    pooled_randomMean = pooled_sum_WistarYi / pooled_sum_Wistar
    pooled_randomVariance = 1 / pooled_sum_Wistar
    pooled_randomStdError = (1 / pooled_sum_Wistar) ** 0.5

    Q_pooled = pooled_sum_WistarYiYi - pooled_sum_WistarYi * pooled_sum_WistarYi / pooled_sum_Wistar
    df_pooled = len(effect_size) - 1
    C_pooled = pooled_sum_Wistar - pooled_sum_WistarWistar / pooled_sum_Wistar

    if study_number == 1:  # Only one study in the meta-analysis results in no between-study variance, so it set to 0.
        pooled_T2 = 0
    elif (Q_pooled - df_pooled) / C_pooled < 0:  # 20210411，Set to 0 if T2 is less than 0.   M.Borenstein[2009] P114
        pooled_T2 = 0
    else:
        pooled_T2 = (Q_pooled - df_pooled) / C_pooled

    if study_number == 1:  # Only one study in the meta-analysis process results in no heterogeneity, so it set to 0.
        pooled_I2 = 0  # the proportion of the observed variance reflects real differences in effect size
    elif ((Q_pooled - df_pooled) / Q_pooled) < 0:  # 20210418，Set to 0 if I2 is less than 0.   M.Borenstein[2009] P110
        pooled_I2 = 0
    else:
        pooled_I2 = ((Q_pooled - df_pooled) / Q_pooled) * 100  # Higgins et al. (2003) proposed using a statistic, I2,

    pooled_pValue_Q = 1.0 - stats.chi2.cdf(Q_pooled, df_pooled)

    d["pooled_mean"] = pooled_randomMean
    d["pooled_Variance"] = pooled_randomVariance
    d["pooled_stdError"] = pooled_randomStdError
    d["pooled_LL_CI"] = pooled_randomMean - 1.96 * pooled_randomStdError  # The 95% lower limits
    d["pooled_L_CI"] = pooled_randomMean + 1.96 * pooled_randomStdError  # The 95% upper limits
    d["pooled_LL_CI_84"] = pooled_randomMean - 1.4051 * pooled_randomStdError  # The 84% lower limits
    d["pooled_UL_CI_84"] = pooled_randomMean + 1.4051 * pooled_randomStdError  # The 84% upper limits
    d["pooled_ZValue"] = pooled_randomMean / pooled_randomStdError  # a Z-value to test the null hypothesis that
    # the mean effect is zero. norm.cdf() 返回标准正态累积分布函数值, 20210414 双侧检验时需要增加绝对值符号np.abs
    d["pooled_pValue_Z"] = 2 * (1 - norm.cdf(np.abs(pooled_randomMean / pooled_randomStdError)))
    d["pooled_Q"] = Q_pooled
    d["pooled_df"] = df_pooled
    d["pooled_C"] = C_pooled
    d["pooled_pValue_Q"] = pooled_pValue_Q
    d["pooled_I2"] = pooled_I2
    d["pooled_tau"] = pooled_T2 ** 0.5
    # tau、randomMean 已知情况下的新出现的study的effect size所落的区间
    d["pooled_LL_ndPred"] = pooled_randomMean - 1.96 * (pooled_T2 ** 0.5)
    d["pooled_UL_ndPred"] = pooled_randomMean + 1.96 * (pooled_T2 ** 0.5)
    # tau、randomMean 未知情况（估计）下的新出现的study的effect size所落的区间
    # stats.t.ppf(0.975,df)返回学生t分布单尾alpha=0.025区间点(双尾是alpha=0.05)的函数，它是stats.t.cdf()累积分布函数的逆函数
    d["pooled_LL_tdPred"] = pooled_randomMean - stats.t.ppf(0.975, pooled_df) \
                            * ((pooled_T2 + pooled_randomStdError * pooled_randomStdError) ** 0.5)
    d["pooled_UL_tdPred"] = pooled_randomMean + stats.t.ppf(0.975, pooled_df) \
                            * ((pooled_T2 + pooled_randomStdError * pooled_randomStdError) ** 0.5)

    # Q-test for separate estimate tau: a Q-test based on analysis of variance, which is identical to Z test and Q-test
    # based on heterogeneity. Q_total = pooled_sum_WiYiYi - pooled_sum_WiYi * pooled_sum_WiYi / pooled_sum_Wistar
    Q_total_separate = separate_sum_WistarYiYi - separate_sum_WistarYi * separate_sum_WistarYi / separate_sum_Wistar
    Q_within_separate = pooled_Q_separate
    Q_between_separate = Q_total_separate - Q_within_separate
    df_between_separate = len(subgroups) - 1
    pValue_Q_between_separate = 1.0 - stats.chi2.cdf(Q_between_separate, df_between_separate)

    d["Q-test_separate_ANOVA"] = pValue_Q_between_separate
    d["Q-test_separate_Q_total"] = Q_total_separate
    d["Q-test_separate_Q_within"] = Q_within_separate
    d["Q-test_separate_Q_between"] = Q_between_separate
    d["Q-test_separate_df_between"] = df_between_separate
    d["Q-test_separate_pValue_Q_between"] = pValue_Q_between_separate

    # Q-test for pooled estimate tau
    Q_total_pooled = pooled_sum_WistarYiYi - pooled_sum_WistarYi * pooled_sum_WistarYi / pooled_sum_Wistar
    Q_within_pooled = pooled_Q_pooled
    Q_between_pooled = Q_total_pooled - Q_within_pooled
    df_between_pooled = len(subgroups) - 1
    pValue_Q_between = 1.0 - stats.chi2.cdf(Q_between_pooled, df_between_pooled)

    d["Q-test_pooled_ANOVA"] = pValue_Q_between
    d["Q-test_pooled_Q_total"] = Q_total_pooled
    d["Q-test_pooled_Q_within"] = Q_within_pooled
    d["Q-test_pooled_Q_between"] = Q_between_pooled
    d["Q-test_pooled_df_between"] = df_between_pooled
    d["Q-test_pooled_pValue_Q_between"] = pValue_Q_between

    # Compute the separate estimate tau-squared：M.Borenstein[2009] P179 did not report the statistic
    separate_Q = Q_total_pooled
    separate_df = len(effect_size) - 1
    separate_C = separate_sum_Wistar - separate_sum_WistarWistar / separate_sum_Wistar
    separate_T2 = (separate_Q - separate_df) / separate_C  # sample estimate of tau squared

    if separate_T2 < 0:
        separate_T2 = 0  # 20210411，Set to 0 if T2 is less than 0.   M.Borenstein[2009] P114
    separate_randomMean = separate_sum_WistarYi / separate_sum_Wistar
    separate_randomVariance = 1 / separate_sum_Wistar
    separate_randomStdError = (1 / separate_sum_Wistar) ** 0.5
    separate_I2 = ((pooled_Q - pooled_df) / pooled_Q) * 100  # Higgins et al. (2003) proposed using a statistic, I2,
    # the proportion of the observed variance reflects real differences in effect size
    if separate_I2 < 0:
        separate_I2 = 0  # 20210418，Set to 0 if I2 is less than 0.   M.Borenstein[2009] P110

    separate_pValue_Q = 1.0 - stats.chi2.cdf(separate_Q, separate_df)

    d["separate_mean"] = separate_randomMean
    d["separate_Variance"] = separate_randomVariance
    d["separate_stdError"] = separate_randomStdError
    d["separate_LL_CI"] = separate_randomMean - 1.96 * separate_randomStdError  # The 95% lower limits
    d["separate_UL_CI"] = separate_randomMean + 1.96 * separate_randomStdError  # The 95% upper limits
    d["separate_LL_CI_84"] = separate_randomMean - 1.4051 * separate_randomStdError  # The 84% lower limits
    d["separate_UL_CI_84"] = separate_randomMean + 1.4051 * separate_randomStdError  # The 84% upper limits
    d["separate_ZValue"] = separate_randomMean / separate_randomStdError  # a Z-value to test the null hypothesis that
    # the mean effect is zero. norm.cdf() 返回标准正态累积分布函数值, 20210414 双侧检验时需要增加绝对值符号np.abs
    d["separate_pValue_Z"] = 2 * (1 - norm.cdf(np.abs(separate_randomMean / separate_randomStdError)))
    d["separate_Q"] = separate_Q
    d["separate_df"] = separate_df
    d["separate_C"] = separate_C
    d["separate_pValue_Q"] = separate_pValue_Q
    d["separate_I2"] = separate_I2
    d["separate_tau"] = separate_T2 ** 0.5
    # tau、randomMean 已知情况下的新出现的study的effect size所落的区间
    d["separate_LL_ndPred"] = separate_randomMean - 1.96 * (separate_T2 ** 0.5)
    d["separate_UL_ndPred"] = separate_randomMean + 1.96 * (separate_T2 ** 0.5)
    # tau、randomMean 未知情况（估计）下的新出现的study的effect size所落的区间
    # stats.t.ppf(0.975,df)返回学生t分布单尾alpha=0.025区间点(双尾是alpha=0.05)的函数，它是stats.t.cdf()累积分布函数的逆函数
    d["separate_LL_tdPred"] = separate_randomMean - stats.t.ppf(0.975, separate_df) \
                              * ((separate_T2 + separate_randomStdError * separate_randomStdError) ** 0.5)
    d["separate_UL_tdPred"] = separate_randomMean + stats.t.ppf(0.975, separate_df) \
                              * ((separate_T2 + separate_randomStdError * separate_randomStdError) ** 0.5)

    # Quantify the magnitude of the difference and comparing each subgroup in Z-test P168
    for s_Z in subgroups:

        non_subgroup = [s_Z]  # exclude the current subgroup

        def fun_1(m):
            return m if m not in non_subgroup else None

        non_s_Z_subgroups = filter(fun_1, subgroups)

        for non_s_Z in non_s_Z_subgroups:
            print("the non_s_Z is ", non_s_Z, "the current subgroup is ", s_Z, " the all subgroups are ", subgroups)
            # Z-test for separate estimate tau: a Z-value to test the null hypothesis that the mean effect is zero
            Diff_star_separate = d["separate_" + non_s_Z + "_mean"] - d["separate_" + s_Z + "_mean"]
            SE_Diff_star_separate = (d["separate_" + non_s_Z + "_variance"] + d["separate_" + s_Z + "_variance"]) ** 0.5

            d["Z_test_separate_" + non_s_Z + "_" + s_Z + "_Diff"] = Diff_star_separate
            d["Z_test_separate_" + non_s_Z + "_" + s_Z + "_SE_Diff"] = SE_Diff_star_separate
            d["Z_test_separate_" + non_s_Z + "_" + s_Z + "_Z"] = Diff_star_separate / SE_Diff_star_separate
            d["Z_test_separate_" + non_s_Z + "_" + s_Z + "_pValue_Z"] = 2 * (
                    1 - norm.cdf(np.abs(Diff_star_separate / SE_Diff_star_separate)))

            # Z-test for pooled estimate tau
            Diff_star_pooled = d["pooled_" + non_s_Z + "_mean"] - d["pooled_" + s_Z + "_mean"]
            SE_Diff_star_pooled = (d["pooled_" + non_s_Z + "_variance"] + d["pooled_" + s_Z + "_variance"]) ** 0.5

            d["Z_test_pooled_" + non_s_Z + "_" + s_Z + "_Diff"] = Diff_star_pooled
            d["Z_test_pooled_" + non_s_Z + "_" + s_Z + "_SE_Diff"] = SE_Diff_star_pooled
            d["Z_test_pooled_" + non_s_Z + "_" + s_Z + "_Z"] = Diff_star_pooled / SE_Diff_star_pooled
            d["Z_test_pooled_" + non_s_Z + "_" + s_Z + "_pValue_Z"] = 2 * (
                    1 - norm.cdf(np.abs(Diff_star_pooled / SE_Diff_star_pooled)))

    return d


# inverse Fisher Transformation
def inverse_Fisher_Z(fisher_Z):
    import numpy as np
    rp = (np.exp(2 * fisher_Z) - 1) / (np.exp(2 * fisher_Z) + 1)
    return rp


def AUC_subgroup_meta_analysis(working_dir="F:\\NJU\\subMeta\\experiments\\subgroupMetaAnalysis\\"):
    import os
    import csv
    import scipy
    import numpy as np
    import pandas as pd
    from sklearn.metrics import recall_score, precision_score, f1_score, roc_curve, auc, roc_auc_score, confusion_matrix

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 5000)

    os.chdir(working_dir)
    print("This is beginning! The current directory is ", os.getcwd())

    with open(working_dir + "AUC_subgroups_MetaAnalysis.csv", 'a+', encoding="utf-8", newline='') as f_AUC, \
            open(working_dir + "Pearson_subgroups_MetaAnalysis.csv", 'a+', encoding="utf-8", newline='') as f_Pearson:

        writer_AUC = csv.writer(f_AUC)
        writer_Pearson = csv.writer(f_Pearson)

        df = pd.read_csv(working_dir + "AUC_MetaAnalysis_Data.csv")
        # drop all rows that have any NaN values,删除表中含有任何NaN的行,并重新设置行号
        df = df.dropna(axis=0, how='any', inplace=False).reset_index(drop=True)

        metric_names = sorted(set(df.metric.values.tolist()))

        subgroup_names = sorted(set(df.subGroup.values.tolist()))
        print(subgroup_names)
        if os.path.getsize(working_dir + "AUC_subgroups_MetaAnalysis.csv") == 0:
            writer_AUC.writerow(["metric"])

        if os.path.getsize(working_dir + "Pearson_subgroups_MetaAnalysis.csv") == 0:
            writer_Pearson.writerow(["metric", "direction_separated", "Pearson_separated_tau",
                                     "Pearson_separated_tau_stdError", "Pearson_separated_tau_variance",
                                     "separate_LL_CI", "separate_UL_CI", "separate_ZValue", "separate_pValue_Z",
                                     "separate_Q"])

        for metric in metric_names:

            print("the current metric is ", metric)

            FisherZ_effect_size = df[df["metric"] == metric].loc[:, "Fisher_Z"].astype(float)
            FisherZ_variance = df[df["metric"] == metric].loc[:, "Fisher_Z_variance"].astype(float)
            FisherZ_subgroup = df[df["metric"] == metric].loc[:, "subGroup"]

            metaThreshold = pd.DataFrame()
            metaThreshold['EffectSize'] = FisherZ_effect_size
            metaThreshold['Variance'] = FisherZ_variance
            metaThreshold['Subgroup'] = FisherZ_subgroup
            # print(metric, FisherZ_effect_size, FisherZ_variance, FisherZ_subgroup)
            # print(metric, np.array(metaThreshold.loc[:, "EffectSize"]), np.array(metaThreshold.loc[:, "Variance"]),
            #       np.array(metaThreshold.loc[:, "Subgroup"]))
            try:
                subgroup_results = subgroup_random_effect_meta_analysis(np.array(metaThreshold.loc[:, "EffectSize"]),
                                                                        np.array(metaThreshold.loc[:, "Variance"]),
                                                                        np.array(metaThreshold.loc[:, "Subgroup"]))
                # d["LL_CI"] = randomMean - 1.96 * randomStdError  # The 95% lower limits for the summary effect
                # d["UL_CI"] = randomMean + 1.96 * randomStdError  # The 95% upper limits for the summary effect
                for s in subgroup_results:
                    print(s, subgroup_results[s])

                meta_stdError = (inverse_Fisher_Z(subgroup_results["separate_UL_CI"])
                                 - inverse_Fisher_Z(subgroup_results["separate_LL_CI"])) / (1.96 * 2)

                # adjusted_result = trimAndFill(np.arrady(metaThreshold.loc[:, "EffectSize"]),
                #                               np.array(metaThreshold.loc[:, "Variance"]), 0)
                # meta_stdError_adjusted = (inverse_Fisher_Z(adjusted_result["UL_CI"])
                #                           - inverse_Fisher_Z(adjusted_result["LL_CI"])) / (1.96 * 2)
                if subgroup_results["separate_pValue_Q"] > 0.5:
                    direction_separate = 0
                else:
                    if inverse_Fisher_Z(subgroup_results["separate_mean"]) > 0:
                        direction_separate = 1
                    else:
                        direction_separate = -1

                writer_Pearson.writerow([metric, direction_separate,
                                         inverse_Fisher_Z(subgroup_results["separate_mean"]), meta_stdError,
                                         meta_stdError * meta_stdError,
                                         inverse_Fisher_Z(subgroup_results["separate_LL_CI"]),
                                         inverse_Fisher_Z(subgroup_results["separate_UL_CI"]),
                                         subgroup_results["separate_ZValue"], subgroup_results["separate_pValue_Z"],
                                         subgroup_results["separate_Q"]])

                #  print the results of each subgroup
                for s in subgroup_names:
                    with open(working_dir + "Pearson_" + s + "_subgroup_MetaAnalysis.csv", 'a+', encoding="utf-8",
                              newline='') as s_Pearson:
                        writer_s_Pearson = csv.writer(s_Pearson)
                        if os.path.getsize(working_dir + "Pearson_" + s + "_subgroup_MetaAnalysis.csv") == 0:
                            writer_s_Pearson.writerow(["metric", "direction_separated", "Pearson_separated_tau",
                                                       "Pearson_separated_tau_stdError",
                                                       "Pearson_separated_tau_variance",
                                                       "separate_LL_CI", "separate_UL_CI", "separate_ZValue",
                                                       "separate_pValue_Z",
                                                       "separate_Q"])
                        meta_s_stdError = (inverse_Fisher_Z(subgroup_results["separate_" + s + "_UL_CI"])
                                         - inverse_Fisher_Z(subgroup_results["separate_" + s + "_LL_CI"])) / (1.96 * 2)
                        if subgroup_results["separate_" + s + "_pValue_Q"] > 0.5:
                            direction_s_separate = 0
                        else:
                            if inverse_Fisher_Z(subgroup_results["separate_" + s + "_mean"]) > 0:
                                direction_s_separate = 1
                            else:
                                direction_s_separate = -1
                        writer_s_Pearson.writerow([metric, direction_s_separate,
                                                   inverse_Fisher_Z(subgroup_results["separate_" + s + "_mean"]),
                                                   meta_s_stdError, meta_s_stdError * meta_s_stdError,
                                                   inverse_Fisher_Z(subgroup_results["separate_" + s + "_LL_CI"]),
                                                   inverse_Fisher_Z(subgroup_results["separate_" + s + "_UL_CI"]),
                                                   subgroup_results["separate_" + s + "_ZValue"],
                                                   subgroup_results["separate_" + s + "_pValue_Z"],
                                                   subgroup_results["separate_" + s + "_Q"]])
            except Exception as err1:
                print(err1)
            break


if __name__ == '__main__':
    import os
    import sys
    import time

    working_directory = "F:\\NJU\\subMeta\\experiments\\subgroupPearson\\"
    s_time = time.time()

    AUC_subgroup_meta_analysis()

    e_time = time.time()
    execution_time = e_time - s_time

    print("The __name__ is ", __name__, ". This", os.path.basename(sys.argv[0]), "ended within", execution_time, "(s).")
