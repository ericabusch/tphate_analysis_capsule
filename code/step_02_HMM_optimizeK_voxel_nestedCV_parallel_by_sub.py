# step_02_HMM_optimizeK_voxel_nestedCV_parallel_by_sub.py
"""
This script begins the HMM event segmentation analysis.
It uses the voxel resolution data for all subjects in a given region and dataset
and uses nested leave-one-subject-out cross validation to optimize the number
of neural events experienced by that region. The best K for a region is chosen as the
K that minimizes the log-likelihood of model fit on the left-out subject, and is then applied
to a validation subject.

As this analysis performs nested leave-one-subject-out cross-validation (repeating the procedure
N * (N-1) times), this script is parallelized at the validation subject level with DSQ.

This K is then held constant for each region across embedding methods for the validation subject.

Runs from the command line as 
`python step_02_HMM_optimizeK_voxel_nestedCV_parallel_by_sub.py $DATASET $ROI $VALIDATION_SUBJECT`
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

def innerloop_optimize_K_voxel(train_data, test_data):
    log_likelihoods = np.zeros_like(K2Test)
    # test all the K values
    for i,K in enumerate(K2Test):
        HMM = brainiak.eventseg.event.EventSegment(K)
        HMM.fit(train_data)
        _, ll = HMM.find_events(test_data)
        log_likelihoods[i] = ll
    # find the best K value
    bestK = K2Test[np.argmax(log_likelihoods)]
    return bestK, np.max(log_likelihoods)

def outerloop_selectK_validation(K_tuning_all_folds_df):
    bestK_dataset = pd.DataFrame(columns=['ROI', 'dataset', 'subject', 'bestK'])
    K_tuning_all_folds_df = K_tuning_all_folds_df[['test_subject', 'validation_subject', 'K']]
    bestK_df = K_tuning_all_folds_df.groupby(["validation_subject"]).mean().reset_index()
    # now need to turn to int
    bestK_df = bestK_df.round({'K':0})
    bestK_df['bestK'] = bestK_df['K']
    return bestK_df[['validation_subject','bestK']]



def run_loops(train_idx, test_idx, validation_idx):
    # take the mean timeseries for the training subjects
    train_data = np.mean(all_subjects_data[train_idx],axis=0)
    test_data = np.squeeze(all_subjects_data[test_idx])
    val_data = np.squeeze(all_subjects_data[validation_idx])
    test_subject = SUBJECTS[test_idx]
    validation_subject = SUBJECTS[validation_idx]
    bestK, test_ll = innerloop_optimize_K_voxel(train_data, test_data)
    if verbose: print(f'finished loop for {test_subject}, {validation_subject}; K={bestK}, testll={test_ll}')
    return bestK, validation_subject, test_subject



def create_loops():
    n_subjects, nTRs, n_features = all_subjects_data.shape
    LOO_outer = LeaveOneOut()
    joblist = []

    overall_df = pd.DataFrame(columns=['ROI', 'dataset', 'test_subject', 'validation_subject', 'K_tested'])
    
    # input arguments holds out one subject for validation parameters (outerloop at script level)  
    # create a second inner loop to hold one subject out of the training set as a testing subj
    train_idx_outer = np.setdiff1d(np.arange(n_subjects), validation_idx)
    for test_idx in train_idx_outer:
        train_idx_inner = np.setdiff1d(train_idx_outer, test_idx)
        if verbose: print(validation_idx, test_idx, train_idx_inner)
        joblist.append(delayed(run_loops)(train_idx_inner, test_idx, validation_idx))
    print(f'starting {len(joblist)} jobs')
    with Parallel(n_jobs=NJOBS) as parallel:
        results = parallel(joblist)

    for r in results:
        bestK, validation_subject, test_subject = r
        temp = {'ROI':ROI, 'dataset':DATASET, 'test_subject':test_subject, 
            'validation_subject':validation_subject, 'K_tested':bestK}
        overall_df.loc[len(overall_df)] = temp

    # now need to groupby validation subject to get an overall mean across test folds
    bestK_df = outerloop_selectK_validation(overall_df)
    bestK_df['ROI'] = [ROI for i in range(len(bestK_df))]
    bestK_df["dataset"]= [DATASET for i in range(len(bestK_df))]
    return bestK_df, overall_df


if __name__ == "__main__":
    verbose = True
    DATASET = sys.argv[1]
    ROI = sys.argv[2]
    if len(sys.argv) > 3:
        validation_idx = int(sys.argv[3])
    else:
        validation_idx=0 # set this for the demo
    
    if verbose: print(DATASET, ROI, validation_idx, NJOBS)
        
        
    SUBJECTS=config.SUBJECTS[DATASET]
    NTPTS=config.TIMEPOINTS[DATASET]
    LOADFN=utils.LOAD_FMRI_FUNCTIONS[DATASET]
    BASE_DIR=config.DATA_FOLDERS_CAPSULE[DATASET]
    DATA_DIR = f'{BASE_DIR}/demo_ROI_data' if DATASET == 'demo' else f'{BASE_DIR}/ROI_data/{ROI}/data'
    K2Test = config.HMM_K_TO_TEST[DATASET]
    OUT_DIR = config.INTERMEDIATE_DATA_FOLDERS[DATASET]+'/HMM_learnK_nested'
    os.makedirs(OUT_DIR, exist_ok=True)
    validation_subject = SUBJECTS[validation_idx]
    
    bestK_df_fn = f'{OUT_DIR}/{DATASET}_{ROI}_bestK_nestedCV_validation_sub-{validation_subject:02d}.csv'
    learnK_df_fn = f'{OUT_DIR}/{DATASET}_{ROI}_samplingK_nestedCV_validation_sub-{validation_subject:02d}.csv'
    try:
        overall_df = pd.read_csv(learnK_df_fn)
        print(f"loaded {learnK_df_fn}")
        bestK_df = outerloop_selectK_validation(overall_df)
        bestK_df['ROI'] = [ROI for i in range(len(bestK_df))]
        bestK_df["dataset"]= [DATASET for i in range(len(bestK_df))]
        print(f"saving {bestK_df_fn}")
    except:
        print(f"Running loops")
        all_subjects_data = np.array(LOADFN(ROI))
        print(all_subjects_data.shape, bestK_df_fn)
        bestK_df, overall_df = create_loops()
    bestK_df.to_csv(bestK_df_fn)
    overall_df.to_csv(learnK_df_fn)

