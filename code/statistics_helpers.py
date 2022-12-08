from random import choices, choice
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

def permutation_test(data, n_iterations, alternative='greater'):
    """
    permutation test for comparing the means of two distributions 
    where the samples between the two distributions are paired
    
    """
    
    def less(null_distribution, observed):
        cmps = null_distribution <= observed + gamma
        pvalues = (cmps.sum(axis=0) + 1) / (n_iterations + 1)
        return pvalues

    def greater(null_distribution, observed):
        cmps = null_distribution >= observed - gamma
        pvalues = (cmps.sum(axis=0) + 1) / (n_iterations + 1)
        return pvalues

    def two_sided(null_distribution, observed):
        pvalues_less = less(null_distribution, observed)
        pvalues_greater = greater(null_distribution, observed)
        pvalues = np.minimum(pvalues_less, pvalues_greater) * 2
        return pvalues
    
    compare = {'less': less, 'greater': greater, 'two-sided': two_sided}
    n_samples = data.shape[1]
    observed_difference = data[0] - data[1]
    observed = np.mean(observed_difference)
    
    
    null_distribution = np.empty(n_iterations)
    for i in range(n_iterations):
        weights = [choice([-1, 1]) for d in range(n_samples)]
        null_distribution[i] = (weights*observed_difference).mean()
        
    eps = 1e-14
    gamma = np.maximum(eps, np.abs(eps * observed))
    
    pvalue = compare[alternative](null_distribution, observed)
    return observed, pvalue, null_distribution

def correct_pvalue(uncorrected_pvalue, n_comparisons):
    return uncorrected_pvalue / n_comparisons
    