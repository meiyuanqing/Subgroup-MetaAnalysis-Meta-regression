#!/usr/bin/env python
# encoding:utf-8
"""
Author : Yuanqing Mei
Date : 2021/9/7
Time: 8:46
File: AUC_subgroup_meta.py
HomePage : http://github.com/yuanqingmei
Email : dg1533019@smail.nju.edu.cn

Inputs: the Pearson (transformed by Z_fisher) and AUC each metric for each subgroup .

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


def AUC_subgroup_meta_analysis(working_directory):
    print("this is beginning!")


if __name__ == '__main__':
    import os
    import sys
    import time

    working_directory = "F:\\NJU\\subMeta\\experiments\\subgroupPearson\\"
    s_time = time.time()

    AUC_subgroup_meta_analysis(working_directory)

    e_time = time.time()
    execution_time = e_time - s_time

    print("The __name__ is ", __name__, ". This", os.path.basename(sys.argv[0]), "ended within", execution_time, "(s).")
