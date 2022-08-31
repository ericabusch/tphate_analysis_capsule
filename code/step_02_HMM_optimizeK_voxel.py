# step_02_HMM_optimizeK_voxel.py
"""
This script begins the HMM event segmentation analysis.
It uses the voxel resolution data for all subjects in a given region and dataset
and uses leave-one-subject-out cross validation to optimize the number
of neural events experienced by that region. The best K for a region is chosen as the
K that minimizes the log-likelihood of model fit on the left-out subject.

This K is then held constant for each region across embedding methods.

Runs from the command line as python step_02_optimizeK_voxel.py $DATASET $ROI
Saves two csv files to an intermediate data directory, for this region and dataset:
    1. selectK_results:
    - saves the results of HMM fitting with different K for each test subject, used to
    determine the best K

    2. bestK_dataset:
    - saves K for each subject, as well as the CV_K to be used in subsequent analyses



"""

import numpy as np
import pandas as pd
import utils, config
import os, sys, glob, pickle
import brainiak.eventseg.event
from scipy.stats import zscore
from sklearn.model_selection import LeaveOneOut, KFold
from joblib import Parallel, delayed
from config import NJOBS

def innerloop_optimize_K_voxel(train_data, test_data, test_subject):
    LL_df = pd.DataFrame(columns=['K', 'LogLikelihood', 'method', 'test_subject', 'ROI', 'dataset'])
    for K in K2Test:
        HMM = brainiak.eventseg.event.EventSegment(K)
        HMM.fit(train_data)
        _, ll = HMM.find_events(test_data)
        LL_df.loc[len(LL_df)] = {'K': K,
                                 'method': 'voxel',
                                 'LogLikelihood': ll,
                                 'ROI': ROI,
                                 'dataset': DATASET,
                                 'test_subject': test_subject}
    bestK = get_bestK(LL_df)
    return bestK, test_subject, LL_df

def outerloop_selectK(all_subjects_data):
    n_subjects, nTRs, n_features = all_subjects_data.shape
    # test on a single subject
    LOO = LeaveOneOut()
    bestK_dataset = pd.DataFrame(columns=['ROI', 'dataset', 'test_subject', 'bestK'])
    joblist = []
    for split, (train_indices, test_index) in enumerate(LOO.split(np.arange(n_subjects))):
        test_subject = SUBJECTS[test_index][0]
        train_data, test_data = all_subjects_data[train_indices], all_subjects_data[test_index]
        train_data, test_data = np.mean(train_data, axis=0), np.squeeze(test_data)
        joblist.append(delayed(innerloop_optimize_K_voxel)(train_data, test_data, test_subject))

    with Parallel(n_jobs=NJOBS) as parallel:
        results = parallel(joblist)

    selectK_results = []
    for r in results:
        temp = {'ROI': ROI,
                'dataset': DATASET,
                'test_subject': r[1],
                'bestK': r[0]}
        bestK_dataset.loc[len(bestK_dataset)] = temp
        selectK_results.append(r[2])  # the dataframe that went into the source
    selectK_results = pd.concat(selectK_results)
    # add in the cross-validated best K results for downstream analyses
    bestK_dataset['CV_K']=np.zeros(len(bestK_dataset))
    for test_subject in SUBJECTS:
        train_subjects=np.setdiff1d(SUBJECTS, test_subject)
        k = int(np.round(bestK_dataset[bestK_dataset['test_subject'].isin(train_subjects)]['bestK'].values.mean()))
        idx = bestK_dataset.index[bestK_dataset['test_subject']==subject]
        bestK_dataset.at[idx, 'CV_K'] = k

    return bestK_dataset, selectK_results

# get the K value resulting in the best LL of model fit
def get_bestK(bestK_df):
    a = bestK_df.LogLikelihood.max()
    bestK = int(bestK_df[bestK_df['LogLikelihood'] == a]['K'].values[0])
    return bestK


if __name__ == "__main__":
    DATASET = sys.argv[1]
    ROI = sys.argv[2]
    SUBJECTS=config.SUBJECTS[DATASET]
    NTPTS=config.TIMEPOINTS[DATASET]
    LOADFN=utils.LOAD_FMRI_FUNCTIONS[DATASET]
    BASE_DIR=config.DATA_FOLDERS[DATASET]
    DATA_DIR = f'{BASE_DIR}/demo_ROI_data' if DATASET == 'demo' else f'{BASE_DIR}/ROI_data/{ROI}/data'
    K2Test = config.HMM_K_TO_TEST[DATASET]
    OUT_DIR = INTERMEDIATE_DATA_FOLDERS[DATASET]+'/HMM_learnK'
    bestK_df_fn = f'{OUT_DIR}/{DATASET}_{ROI}_bestK_LOSO.csv'
    learnK_df_fn = f'{OUT_DIR}/{DATASET}_{ROI}_samplingK_LOSO.csv'

    data = np.array(LOADFN(ROI))
    bestK_dataset, selectK_results = outerloop_selectK(data)
    bestK_dataset.to_csv(bestK_df_fn)
    selectK_results.to_csv(learnK_df_fn)

