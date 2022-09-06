"""
step_06_WvB_boundaries_cross_subjects.py

This script runs the analyses presented in Figure 6. Using the event boundaries identified using
HMMs within-subject that were saved from step 3 and step 4, this script evaluates the fit of these
boundaries across subjects. In other words, for a subject N, it iteratively takes the boundaries identified
by HMMs fit on all other N-1 subjects and evaluates how well they fit subject N's data, using the same
metric of boundary fit.


Runs from the command line as:
python step_06_WvB_boundaries_cross_subjects.py $DATASET $ROI $METHOD

"""

import numpy as np
import pandas as pd
import utils, config
import os, sys, glob
from scipy.stats import zscore
from joblib import Parallel, delayed
import embedding_helpers as mf
from scipy.spatial.distance import pdist, cdist, squareform
from step_03_HMM_optimizeM_embeddings import compute_event_boundaries_diff_temporally_balanced
from config import NJOBS

## Tests the fit of event boundaries from each of N-1 subjects on the Nth subject
def run_single_subject(data, training_subject_boundaries, test_subject_ID):
    training_subject_IDs = np.setdiff1d(SUBJECTS, test_subject_ID)
    df_here = pd.DataFrame(columns=["test_subject", "train_subject", "avg_diff", "avg_between", "avg_within", "K"])
    for train_sub, boundaries in zip(training_subject_IDs, training_subject_boundaries):
        diffs, withins, betweens, _ = compute_event_boundaries_diff_temporally_balanced(data, boundaries)
        df_here.loc[len(df_here)] = {"test_subject": test_subject_ID,
                                     "train_subject": train_sub,
                                     "avg_diff": np.nanmean(diffs),
                                     "avg_between": np.nanmean(betweens),
                                     "avg_within": np.nanmean(withins),
                                     "K": len(boundaries) - 1}
    return df_here


def load_embeddings(subject_IDs, CV_Ms):
    embeddings = []
    for sub, M in zip(subject_IDs, CV_Ms):
        embed_fn = os.path.join(EMBED_DIR, f'sub-{sub:02d}_{ROI}_{SEARCHSTR}_{M}dimension_embedding_{METHOD}.npy')
        embeddings.append(np.load(embed_fn))
    return embeddings


def main():

    if METHOD == 'voxel':
        allsubj_data = LOADFN(ROI)
    else:
        CV_Ms = param_df['CV_M'].values
        allsubj_data = load_embeddings(SUBJECTS, CV_Ms)
    # define some functions for handling string parsing
    formatList = lambda string: string.replace('[', '').replace(']', '').split(', ')
    formatArr = lambda myList: [int(i) for i in myList]
    formatAll = lambda string: formatArr(formatList(string))
    # parse from the dataframe that you stored dumbly
    boundary_TRs_formatted = []
    subset_bounds = param_df['boundary_TRs'].values
    for i, boundaries in enumerate(subset_bounds):
        arr = formatAll(boundaries)
        boundary_TRs_formatted.append(np.array(arr))
    
    joblist = []
    for sub_idx, sub in enumerate(SUBJECTS):
        dat = allsubj_data[sub_idx]
        train_subject_idx = np.setdiff1d(np.arange(len(SUBJECTS)), sub_idx)
        train_subj_boundaries = [boundary_TRs_formatted[i] for i in train_subject_idx]
        joblist.append(delayed(run_single_subject)(dat, train_subj_boundaries, sub))

    print("starting jobs")
    with Parallel(n_jobs=NJOBS) as parallel:
        results_df_list = parallel(joblist)
    print(results_df_list)
    results_df = pd.concat(results_df_list)
    results_df['ROI'] = np.repeat(ROI, len(results_df))
    results_df['embed_method'] = np.repeat(METHOD, len(results_df))
    results_df['dataset'] = np.repeat(DATASET, len(results_df))
    results_df.to_csv(outfn)
    print(f'saved to {outfn}')


if __name__ == '__main__':
    DATASET = sys.argv[1]
    ROI = sys.argv[2]
    METHOD = sys.argv[3]
    BASE_DIR = config.DATA_FOLDERS[DATASET]
    LOADFN = utils.LOAD_FMRI_FUNCTIONS[DATASET]
    DATA_DIR = f'{BASE_DIR}/demo_ROI_data' if DATASET == 'demo' else f'{BASE_DIR}/ROI_data/{ROI}/data'
    EMBED_DIR = f'{BASE_DIR}/demo_embeddings' if DATASET == 'demo' else f'{BASE_DIR}/ROI_data/{ROI}/embeddings'
    OUT_DIR = config.RESULTS_FOLDERS[DATASET]
    SEARCHSTR = config.FILE_STRINGS[DATASET]
    param_df = pd.read_csv(f"{OUT_DIR}/within_sub_neural_event_WB_tempBalance_results.csv", index_col=0)
    D = "sherlock" if DATASET == "demo" else DATASET
    param_df = param_df[(param_df['dataset'] == D) & (param_df['ROI'] == ROI)
                        & (param_df['embed_method'] == METHOD)]
    SUBJECTS = param_df['subject'].values
    outfn = f'{OUT_DIR}/source/{ROI}_{DATASET}_{METHOD}_between_sub_neural_event_WB_tempBalance.csv'
    main()

