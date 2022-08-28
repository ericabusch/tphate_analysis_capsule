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
import utils
from joblib import Parallel, delayed
from scipy.stats import zscore
from step_03_HMM_optimizeM_embeddings import test_boundaries_corr_diff


def main():
    global SUBJECTS
    bestK_df_fn = f'{output_dir}/{DATASET}_{ROI}_bestK_LOSO.csv'
    outdf_fn = f'{results_dir}/{ROI}_{DATASET}_voxel_tempBalance_crossValid_WB_results.csv'
    bestK_df = pd.read_csv(bestK_df_fn, index_col=0)  # this has the best K when holding out each subject
    bestK_df = bestK_df[(bestK_df['ROI'] == ROI)]
    K_by_subject = bestK_df['CV_K'].values
    SUBJECTS = list(SUBJECTS)

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
    print('finished parallel')
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

    if len(sys.argv) > 3:
        print("Running demo; adjusting parameters accordingly")
        DATASET = 'sherlock'
        ROI = 'early_visual'
        SUBJECTS = utils.sherlock_subjects
        NTPTS = utils.sherlock_timepoints
        LOADFN = utils.load_sherlock_movie_ROI_data
        base_dir = "../data/demo_data/"
        data_dir = f'{base_dir}/demo_ROI_data/'
        LOADFN = utils.load_demo_data
        embed_dir = f'{base_dir}/demo_embeddings/'
        output_dir = '../intermediate_data/demo/HMM_learnK'
        results_dir = '../results/demo_results/source/'
    else:
        DATASET = sys.argv[1]
        ROI = sys.argv[2]
        SUBJECTS = utils.sherlock_subjects if DATASET == 'sherlock' else utils.forrest_subjects
        NTPTS = utils.sherlock_timepoints if DATASET == 'sherlock' else utils.forrest_movie_timepoints
        LOADFN = utils.load_sherlock_movie_ROI_data if DATASET == 'sherlock' else utils.load_forrest_movie_ROI_data
        SUBJECTS = utils.sherlock_subjects if DATASET == 'sherlock' else utils.forrest_subjects
        NTPTS = utils.sherlock_timepoints if DATASET == 'sherlock' else utils.forrest_movie_timepoints
        LOADFN = utils.load_sherlock_movie_ROI_data if DATASET == 'sherlock' else utils.load_forrest_movie_ROI_data
        base_dir = utils.sherlock_dir if DATASET == 'sherlock' else utils.forrest_dir
        embed_dir = f'{base_dir}/ROI_data/{ROI}/embeddings/'
        output_dir = '../intermediate_data/HMM_learnK'
        results_dir = '../results/source/'
    NJOBS=16
    main()