# step_04_HMM_WvB_boundaries_voxel.py
"""
This script is to run the voxel-wise analyses presented in figure 4 and figure 5.
This analysis does basically the same thing as in step_03, but excludes the cross-validation
required to determine the "M" parameter, the voxel-resolution data does not get dimensionality
reduction.
"""


import numpy as np
import pandas as pd
import os, sys, glob
import utils, config
from joblib import Parallel, delayed
from scipy.stats import zscore
from step_03_HMM_optimizeM_embeddings import test_boundaries_corr_diff
from config import NJOBS


def main():
    global SUBJECTS
    bestK_df_fn = f'{output_dir}/{DATASET}_{ROI}_bestK_LOSO.csv'
    outdf_fn = f'{results_dir}/{ROI}_{DATASET}_voxel_tempBalance_crossValid_WB_results.csv'
    bestK_df = pd.read_csv(bestK_df_fn, index_col=0)  # this has the best K when holding out each subject
    bestK_df = bestK_df[(bestK_df['ROI'] == ROI)]
    K_by_subject = bestK_df['CV_K'].values

    data=LOADFN(ROI)
    final_df = pd.DataFrame(
        columns=['subject', 'ROI', 'dataset', 'avg_within_event_corr', 'avg_between_event_corr', 'avg_difference',
                 'CV_M', 'CV_K', 'embed_method', 'boundary_TRs', 'model_LogLikelihood', 'compared_timepoints'])
    joblist, parameters = [], []
    for i, subject in enumerate(SUBJECTS):
        test_data = data[i]
        bestK_cv = int(K_by_subject[i])
        parameters.append({'subject': subject, 'ROI': ROI,
                           'dataset': DATASET, 'CV_M': test_data.shape[1],
                           'CV_K': bestK_cv})
        print(f'subject {subject} voxel dataset {DATASET} roi {ROI}; bestK {bestK_cv}')
        joblist.append(delayed(test_boundaries_corr_diff)(test_data, bestK_cv, balance_distance=True))

    with Parallel(n_jobs=NJOBS) as parallel:
        results = parallel(joblist)
    for p, r in zip(parameters, results):
        avg_within, avg_between, avg_diff, boundary_TRs, ll, comparisons = r

        final_df.loc[len(final_df)] = {'subject': p['subject'],
                                       'ROI': p['ROI'],
                                       'dataset': p['dataset'],
                                       'avg_within_event_corr': avg_within,
                                       'avg_between_event_corr': avg_between,
                                       'avg_difference': avg_diff,
                                       'CV_M': p['CV_M'],
                                       'CV_K': p['CV_K'],
                                       'embed_method': 'voxel',
                                       'boundary_TRs': boundary_TRs,
                                       'model_LogLikelihood': ll,
                                       'compared_timepoints': comparisons}

    final_df.to_csv(outdf_fn)
    print(outdf_fn)






if __name__ == "__main__":
    DATASET = sys.argv[1]
    ROI = sys.argv[2]
    SUBJECTS = config.SUBJECTS[DATASET]
    NTPTS = config.TIMEPOINTS[DATASET]
    LOADFN = utils.LOAD_FMRI_FUNCTIONS[DATASET]
    BASE_DIR = config.DATA_FOLDERS[DATASET]
    DATA_DIR = f'{BASE_DIR}/demo_ROI_data' if DATASET == 'demo' else f'{BASE_DIR}/ROI_data/{ROI}/data'
    K2Test = config.HMM_K_TO_TEST[DATASET]
    OUT_DIR = config.INTERMEDIATE_DATA_FOLDERS[DATASET] + '/HMM_learnK'
    RESULTS_DIR =config.RESULTS_FOLDERS[DATASET]+'/source'

    bestK_df_fn = f'{OUT_DIR}/{DATASET}_{ROI}_bestK_LOSO.csv'
    learnK_df_fn = f'{OUT_DIR}/{DATASET}_{ROI}_samplingK_LOSO.csv'

    main()