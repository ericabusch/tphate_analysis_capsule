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
import brainiak.eventseg.event

from joblib import Parallel, delayed
import embedding_helpers as mf
from scipy.spatial.distance import pdist, cdist, squareform
from step_03_HMM_optimizeM_embeddings import compute_event_boundaries_diff_temporally_balanced
from config import NJOBS

## Tests the fit of event boundaries from each of N-1 subjects on the Nth subject
def run_CV_loop(train_subjects, train_idx, test_subject, test_idx):
    df_here = pd.DataFrame(columns=['test_subject','train_subject','avg_diff','avg_between','avg_within','K','M'])
    
    # get CV hyperparameters
    # this was optimized on this set of training subjects; never saw test subject
    CV_K = int(param_df[param_df['subject'] == test_subject]['CV_K'])
    # this was optimized on this set of training subjects; never saw test subject
    CV_M = int(param_df[param_df['subject'] == test_subject]['CV_M'])
    
    # load all data
    if METHOD == 'voxel':
        allsubj_data = LOADFN(ROI)
    else:
        CV_Ms = param_df['CV_M'].values
        allsubj_data = load_embeddings(SUBJECTS, np.repeat(CV_M, len(SUBJECTS)))
    
    # now we have the data; loop through training subjects
    test_data = allsubj_data[test_idx]
    for train_sub, idx in zip(train_subjects, train_idx):
        train_data = allsubj_data[idx]
        # fit an HMM with those parameters on the training data
        HMM = brainiak.eventseg.event.EventSegment(CV_K)
        HMM.fit(train_data)
        _, ll = HMM.find_events(train_data)
        # extract event boundaries
        boundary_TRs = np.where(np.diff(np.argmax(HMM.segments_[0], axis=1)))[0]
        # apply those boundaries to the test subject
        diffs, withins, betweens, _, _ = compute_event_boundaries_diff_temporally_balanced(test_data, boundary_TRs)
        if config.VERBOSE: print(f'Test={test_subject}, train={train_sub}, diff={np.nanmean(diffs)}')
        df_here.loc[len(df_here)] = {"test_subject": test_subject,
                                     "train_subject": train_sub,
                                     "avg_diff": np.nanmean(diffs),
                                     "avg_between": np.nanmean(betweens),
                                     "avg_within": np.nanmean(withins),
                                     "K": CV_K,
                                    "M":CV_M}
    if config.VERBOSE: print(f"Done {test_subject}")
    return df_here 


def load_embeddings(subject_IDs, CV_Ms):
    embeddings = []
    for sub, M in zip(subject_IDs, CV_Ms):
        embed_fn = os.path.join(EMBED_DIR, f'sub-{sub:02d}_{ROI}_{SEARCHSTR}_{M}dimension_embedding_{METHOD}.npy')
        embeddings.append(np.load(embed_fn))
    if config.VERBOSE: print("Loaded embeddings")
    return embeddings


def main():
    
    joblist = []
    print(SUBJECTS)
    for test_sub_idx, test_subject in enumerate(SUBJECTS):
        train_subjects = np.setdiff1d(SUBJECTS, test_subject)
        train_idx = np.setdiff1d(np.arange(len(SUBJECTS)), test_sub_idx)
        joblist.append(delayed(run_CV_loop)(train_subjects, train_idx, test_subject, test_sub_idx))
        print('appended job')

    print("starting jobs")
    with Parallel(n_jobs=NJOBS) as parallel:
        results_df_list = parallel(joblist)
        
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
    print(f'RUNNING {DATASET} {ROI} {METHOD}')
    
    param_df = param_df[(param_df['dataset'] == D) & (param_df['ROI'] == ROI)
                        & (param_df['embed_method'] == METHOD)]
    SUBJECTS = param_df['subject'].values
    TEMP_DIR = f'{config.INTERMEDIATE_DATA_FOLDERS[DATASET]}/HMM_learnK_nested'
    outfn = f'{OUT_DIR}/source/{ROI}_{DATASET}_{METHOD}_between_sub_neural_event_WB_tempBalance_CV.csv'
    if os.path.exists(outfn):
        print(f"Already ran {outfn}")
        sys.exit()
    main()

