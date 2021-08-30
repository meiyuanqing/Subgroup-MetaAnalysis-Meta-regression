#!/usr/bin/env python
# encoding:utf-8
"""
Author : Yuanqing Mei
Date : 2021/8/30
Time: 18:28
File: pearson_sub_meta.py
HomePage : http://github.com/yuanqingmei
Email : dg1533019@smail.nju.edu.cn

Do Random subgroup Meta-Analysis For Pearson, including sensitive analysis by trim and fill method.
The outputs include the meta-analysis results for cpp, cs, java, c, and pascal subgroup, respectively.
In addition, include the pooled and separated estimate Tau-square for all subgroups.

"""
import time


def pearson_sub_meta(working_dir="F:\\NJU\\subMeta\\experiments\\subgroupPearson\\"):
    import os
    import csv
    from scipy.stats import norm  # norm.cdf() the cumulative normal distribution function in Python
    # Calculation of p-values based on the chi-square distribution: p_value=1.0-stats.chi2.cdf(chisquare,freedom_degree)
    from scipy import stats
    import numpy as np
    import pandas as pd

    # input: two anonymous arrays, namely effect_size stores each study's effect size and its variance
    # output: the results of fixed effect model, including
    #        (1) fixedMean：the average of effect sizes;
    #        (2) fixedStdError: the standard error corresponding to the average of effect sizes.
    def fixed_effect_meta_analysis(effect_size, variance):
        fixed_weight = []
        sum_Wi = 0
        sum_WiWi = 0
        sum_WiYi = 0  # Sum(Wi*Yi), where i ranges from 1 to k, and k is the number of studies
        sum_WiYiYi = 0  # Sum(Wi*Yi*Yi), where i ranges from 1 to k, and k is the number of studies
        d = {}  # return a dict
        study_number = len(variance)
        for i in range(study_number):
            if variance[i] == 0:
                continue
            fixed_weight.append(1 / variance[i])
            sum_Wi = sum_Wi + fixed_weight[i]
            sum_WiYi = sum_WiYi + effect_size[i] * fixed_weight[i]
            sum_WiWi = sum_WiWi + fixed_weight[i] * fixed_weight[i]
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

        fixedMean = sum_WiYi / sum_Wi  # the average of effect sizes
        fixedStdError = (1 / sum_Wi) ** 0.5  # the standard error corresponding to the average of effect sizes
        d['fixedMean'] = fixedMean
        d['fixedStdError'] = fixedStdError
        d['tau'] = T2
        return d

    # input: two anonymous arrays, namely effect_size stores each study's effect size and its variance
    # output: the results of random effects model, including
    #         (1) randomMean：the average of effect sizes;
    #         (2) randomStdError: the standard error corresponding to the average of effect sizes.
    def random_effect_meta_analysis(effect_size, variance):

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
            I2 = 0        # 20210418，Set to 0 if I2 is less than 0.   M.Borenstein[2009] P110

        pValue_Q = 1.0 - stats.chi2.cdf(Q, df)  # pValue_Q = 1.0 - stats.chi2.cdf(chisquare, freedom_degree)

        d["C"] = C
        d["mean"] = randomMean
        d["stdError"] = randomStdError
        d["LL_CI"] = randomMean - 1.96 * randomStdError  # The 95% lower limits for the summary effect
        d["UL_CI"] = randomMean + 1.96 * randomStdError  # The 95% upper limits for the summary effect
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

    # inverse Fisher Transformation
    def inverse_Fisher_Z(fisher_Z):
        rp = (np.exp(2 * fisher_Z) - 1) / (np.exp(2 * fisher_Z) + 1)
        return rp

    def getEstimatedK0(effectSizeArray, mean):
        centeredEffectSizeArray = []
        absoluteCenteredEffectSizeArray = []
        size = len(effectSizeArray)
        for i in range(size):
            centeredEffectSizeArray.append(effectSizeArray[i] - mean)
            absoluteCenteredEffectSizeArray.append(np.abs(effectSizeArray[i] - mean))
        sortedArray = sorted(absoluteCenteredEffectSizeArray)
        rank = {sortedArray[0]: 1}  # return a dict
        initialRankValue = 1
        predValue = sortedArray[0]
        for i in range(size):
            if sortedArray[i] > predValue:
                predValue = sortedArray[i]
                initialRankValue += 1
            rank[sortedArray[i]] = initialRankValue
        finalRank = []
        for i in range(size):
            if centeredEffectSizeArray[i] < 0:
                finalRank.append((-1) * rank[absoluteCenteredEffectSizeArray[i]])
            else:
                finalRank.append(rank[absoluteCenteredEffectSizeArray[i]])
        gamma = finalRank[size - 1] + finalRank[0]
        SumPositiveRank = 0
        for i in range(size):
            if finalRank[i] < 0:
                continue
            SumPositiveRank = SumPositiveRank + finalRank[i]
        R0 = int(gamma + 0.5) - 1
        temp = (4 * SumPositiveRank - size * (size + 1)) / (2 * size - 1)
        L0 = int(temp + 0.5)
        if R0 < 0:
            R0 = 0
        if L0 < 0:
            L0 = 0
        return R0, L0

    # Duval and Tweedie's trim and fill method
    def trimAndFill(effect_size, variance, isAUC):
        effectSizeArray = effect_size
        varianceArray = variance
        size = len(effect_size)
        # 检查是否需要切换方向，因为trim and fill方法假设miss most negative的研究
        flipFunnel = 0
        metaAnalysisForFlip = fixed_effect_meta_analysis(effectSizeArray, varianceArray)
        meanForFlip = metaAnalysisForFlip["fixedMean"]

        tempSorted = sorted(effectSizeArray)
        min = tempSorted[0] - meanForFlip
        max = tempSorted[-1] - meanForFlip

        if np.abs(min) > np.abs(max):
            flipFunnel = 1
            for i in range(size):
                effectSizeArray[i] = (-1) * effectSizeArray[i]

        # 按effect size排序
        merge = []
        for i in range(size):
            merge.append([effect_size[i], variance[i]])
        sortedMerge = sorted(merge)
        OrignalEffectSizeArray = []
        OrignalVarianceArray = []
        for i in range(len(sortedMerge)):
            OrignalEffectSizeArray.append(sortedMerge[i][0])
            OrignalVarianceArray.append(sortedMerge[i][1])
        # 迭代算法，估算k0
        metaAnalysisResult = fixed_effect_meta_analysis(OrignalEffectSizeArray, OrignalVarianceArray)
        mean = metaAnalysisResult["fixedMean"]
        RL = getEstimatedK0(OrignalEffectSizeArray, mean)
        R0 = RL[0]
        L0 = RL[1]
        k0 = L0  # 默认的情况利用L0来估算k0
        if (k0 == 0) or (k0 > size):
            result = random_effect_meta_analysis(effect_size, variance)
            result["k0"] = k0
            return result
        trimmedMean = mean
        change = 1
        count = 0
        while change and (size - k0) > 2 and (count < 1000):
            count += 1
            upperBound = size - k0 - 1
            trimmedEffectSizeArray = []
            trimmedVarianceArray = []
            for i in range(upperBound):
                trimmedEffectSizeArray.append(OrignalEffectSizeArray[i])
                trimmedVarianceArray.append(OrignalVarianceArray[i])
            trimmedMetaAnalysisResult = fixed_effect_meta_analysis(trimmedEffectSizeArray, trimmedVarianceArray)
            trimmedMean = trimmedMetaAnalysisResult["fixedMean"]
            trimmedR0_L0 = getEstimatedK0(OrignalEffectSizeArray, trimmedMean)
            trimmedR0 = trimmedR0_L0[0]
            trimmedL0 = trimmedR0_L0[1]
            k1 = trimmedL0
            if k1 == k0:
                change = 0
            k0 = k1
        filledEffectSizeArray = []
        filledVarianceArray = []

        for j in range(k0):
            imputedEffectSize = 2 * trimmedMean - OrignalEffectSizeArray[size - j - 1]
            imputedVariance = OrignalVarianceArray[size - j - 1]
            filledEffectSizeArray.append(imputedEffectSize)
            filledVarianceArray.append(imputedVariance)
        fullEffectSizeArray = filledEffectSizeArray
        fullVarianceArray = filledVarianceArray
        fullEffectSizeArray.extend(OrignalEffectSizeArray)
        fullVarianceArray.extend(OrignalVarianceArray)
        if flipFunnel:
            newSize = len(fullEffectSizeArray)
            for i in range(newSize):
                fullEffectSizeArray[i] = -1 * fullEffectSizeArray[i]

        if isAUC:
            # AUC应该在0到1之间，否则有错
            filteredFullEffectSizeArray = []
            filteredFullVarianceArray = []
            for i in range(len(fullEffectSizeArray)):
                if fullEffectSizeArray[i] < 0:
                    continue
                if fullEffectSizeArray[i] > 1:
                    continue
                filteredFullEffectSizeArray.append(fullEffectSizeArray[i])
                filteredFullVarianceArray.append(fullVarianceArray[i])
            result = random_effect_meta_analysis(filteredFullEffectSizeArray, filteredFullVarianceArray)
            finalk0 = len(filteredFullEffectSizeArray) - len(OrignalEffectSizeArray)
        else:
            result = random_effect_meta_analysis(fullEffectSizeArray, fullVarianceArray)
            finalk0 = len(fullEffectSizeArray) - len(OrignalEffectSizeArray)
        result["k0"] = finalk0
        result["flipFunnel"] = flipFunnel
        return result

    working_directory = working_dir
    os.chdir(working_directory)
    print(os.getcwd())

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 5000)

    PLs = ["cpp", "cs", "java", "c", "pascal"]

    for PL in PLs:
        print("This is ", PL, " studies!")
        # with open(working_directory + "PearsonEffect\\" + PL + "\\" + PL + "_List.txt") as l:
        #     lines = l.readlines()

        # read_csv(path, keep_default_na=False, na_values=[""])  只有一个空字段将被识别为NaN
        df = pd.read_csv(working_directory + "PearsonEffect\\" + PL + "\\" + "Pearson_effects.csv",
                         keep_default_na=False, na_values=[""])
        df = df.dropna(axis=0, how='any', inplace=False).reset_index(drop=True)

        metric_names = sorted(set(df.metric.values.tolist()))
        print("the metric_names are ", df.columns.values.tolist())
        print("the metric_names are ", metric_names)
        print("the len metric_names are ", len(metric_names))
        k = 0
        for metric in metric_names:

            print("the current metric is ", metric)

            FisherZ_effect_size = df[df["metric"] == metric].loc[:, "Fisher_Z"].astype(float)
            # print("the FisherZ_effect_size items are ", FisherZ_effect_size)
            # print("the type FisherZ_effect_size items are ", type(FisherZ_effect_size))
            # print("the len of FisherZ_effect_size items is ", len(FisherZ_effect_size))

            FisherZ_variance = df[df["metric"] == metric].loc[:, "Fisher_Z_variance"].astype(float)
            # print("the threshold_variance items are ", FisherZ_variance)
            # print("the type threshold_variance items are ", type(FisherZ_variance))
            # print("the len of threshold_variance items is ", len(FisherZ_variance))

            metaThreshold = pd.DataFrame()
            metaThreshold['EffectSize'] = FisherZ_effect_size
            metaThreshold['Variance'] = FisherZ_variance
            try:
                resultMetaAnalysis = random_effect_meta_analysis(np.array(metaThreshold.loc[:, "EffectSize"]),
                                                                 np.array(metaThreshold.loc[:, "Variance"]))

                # d["LL_CI"] = randomMean - 1.96 * randomStdError  # The 95% lower limits for the summary effect
                # d["UL_CI"] = randomMean + 1.96 * randomStdError  # The 95% upper limits for the summary effect
                meta_stdError = (inverse_Fisher_Z(resultMetaAnalysis["UL_CI"])
                                 - inverse_Fisher_Z(resultMetaAnalysis["LL_CI"])) / (1.96 * 2)

                adjusted_result = trimAndFill(np.array(metaThreshold.loc[:, "EffectSize"]),
                                              np.array(metaThreshold.loc[:, "Variance"]), 0)
                meta_stdError_adjusted = (inverse_Fisher_Z(adjusted_result["UL_CI"])
                                 - inverse_Fisher_Z(adjusted_result["LL_CI"])) / (1.96 * 2)
                if resultMetaAnalysis["pValue_Z"] > 0.5:
                    direction = 0
                else:
                    if inverse_Fisher_Z(resultMetaAnalysis["mean"]) > 0:
                        direction = 1
                    else:
                        direction = -1

                with open(working_directory + "Pearson_effects_sub_meta.csv", 'a+', encoding="utf-8", newline='') as f:
                    writer_f = csv.writer(f)
                    if os.path.getsize(working_directory + "Pearson_effects_sub_meta.csv") == 0:
                        writer_f.writerow(
                            ["subgroup", "metric", "Pearson_effects_meta", "Pearson_effects_meta_stdError", "LL_CI",
                             "UL_CI", "ZValue", "pValue_Z", "Q", "df", "pValue_Q", "I2", "tau", "LL_ndPred",
                             "UL_ndPred", "number_of_effect_size",
                             "k_0", "Pearson_effects_meta_adjusted", "Pearson_effects_meta_stdError_adjusted",
                             "LL_CI_adjusted", "UL_CI_adjusted", "direction", "pValue_Z_adjusted", "Q_adjusted",
                             "df_adjusted", "pValue_Q_adjusted", "I2_adjusted", "tau_adjusted", "LL_ndPred_adjusted",
                             "UL_ndPred_adjusted"])
                    writer_f.writerow([PL, metric, inverse_Fisher_Z(resultMetaAnalysis["mean"]), meta_stdError,
                                       inverse_Fisher_Z(resultMetaAnalysis["LL_CI"]),
                                       inverse_Fisher_Z(resultMetaAnalysis["UL_CI"]),
                                       resultMetaAnalysis["ZValue"], resultMetaAnalysis["pValue_Z"],
                                       resultMetaAnalysis["Q"], resultMetaAnalysis["df"], resultMetaAnalysis["pValue_Q"],
                                       resultMetaAnalysis["I2"], resultMetaAnalysis["tau"],
                                       inverse_Fisher_Z(resultMetaAnalysis["LL_ndPred"]),
                                       inverse_Fisher_Z(resultMetaAnalysis["UL_ndPred"]), len(FisherZ_effect_size),
                                       adjusted_result["k0"], inverse_Fisher_Z(adjusted_result["mean"]),
                                       meta_stdError_adjusted, inverse_Fisher_Z(adjusted_result["LL_CI"]),
                                       inverse_Fisher_Z(adjusted_result["UL_CI"]), direction,
                                       adjusted_result["pValue_Z"],
                                       adjusted_result["Q"], adjusted_result["df"], adjusted_result["pValue_Q"],
                                       adjusted_result["I2"], adjusted_result["tau"],
                                       inverse_Fisher_Z(adjusted_result["LL_ndPred"]),
                                       inverse_Fisher_Z(adjusted_result["UL_ndPred"])])

            except Exception as err1:
                print(err1)

            k += 1
            # break


if __name__ == '__main__':
    s_time = time.time()
    pearson_sub_meta()
    e_time = time.time()
    execution_time = e_time - s_time

    print("The __name__ is ", __name__, ". This is end of pearson_sub_meta.py! ",
          "And the elapsed time is ", execution_time, " s.")
