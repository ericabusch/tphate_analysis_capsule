from random import choices
import numpy as np
import pandas as pd

def bootstrap_ci(data1, data2, repetitions = 1000, alpha = 0.05):
    bootstrap_sample_size = len(data1)
    mean_diffs = []
    for i in range(repetitions):
        bootstrap_sample_d1 = choices(data1, k=bootstrap_sample_size)
        bootstrap_sample_d2 = choices(data2, k=bootstrap_sample_size)
        mean_diff = np.mean(bootstrap_sample_d1) - np.mean(bootstrap_sample_d2)
        mean_diffs.append(mean_diff)
    lower = np.percentile(mean_diffs, alpha/2*100)
    upper = np.percentile(mean_diffs, 100-alpha/2*100)
    point_est = data1.mean() - data2.mean()
    return lower, upper, point_est

def run_bootstraps(df, cat2vary, level1, level2, value_str):
    m1_data = df[df[cat2vary] == level1][value_str].values
    m2_data = df[df[cat2vary] == level2][value_str].values
    left, right, point = bootstrap_ci(m1_data, m2_data, repetitions = 1000, alpha = 0.05)
    SE = (right - left)/(2 * 1.96)
    z = point/SE
    P = np.exp(-.717*z - 0.416*(z**2))
    return {'point_estimate':point, 'lower_bound':left, 'upper_bound':right, 'SE':SE, 'p':P}