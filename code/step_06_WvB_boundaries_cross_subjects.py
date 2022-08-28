"""
step_06_WvB_boundaries_cross_subjects.py

This script runs the analyses presented in Figure 6. Using the event boundaries identified using
HMMs within-subject that were saved from step 3 and step 4, this script evaluates the fit of these
boundaries across subjects. In other words, for a subject N, it iteratively takes the boundaries identified
by HMMs fit on all other N-1 subjects and evaluates how well they fit subject N's data, using the same
metric of boundary fit.


Runs from the command line as:
python step_06_WvB_boundaries_cross_subjects.py $DATASET $ROI $METHOD $DEMO
where demo is any additional argument, but runs the demo version.

"""



import numpy as np
import pandas as pd
import utils
import os, sys, glob
from scipy.stats import zscore
from joblib import Parallel, delayed
import embedding_helpers as mf
from scipy.spatial.distance import pdist, cdist, squareform
from step_03_HMM_optimizeM_embeddings import compute_event_boundaries_diff_temporally_balanced


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
        embed_fn = os.path.join(data_dir, f'sub-{sub:02d}_{ROI}_{searchstr}_{M}dimension_embedding_{METHOD}.npy')
        embeddings.append(np.load(embed_fn))
    return embeddings


def main():
    global SUBJECTS
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
    failed = []
    subset_bounds = param_df['boundary_TRs'].values
    for i, boundaries in enumerate(subset_bounds):
        try:
            arr = formatAll(boundaries)
            boundary_TRs_formatted.append(np.array(arr))
        except:
            print(f'failed on {i}')
            failed.append(i)
            continue

    # handle subjects with missing boundaries & remove from subsequent analyses
    if len(failed) != 0:
        mask = []
        A = len(SUBJECTS)
        for j in np.arange(A):
            if j not in failed:
                mask.append(1)
            else:
                mask.append(0)
        SUBJECTS = SUBJECTS[mask]
    print("loaded boundaries", len(boundary_TRs_formatted), SUBJECTS)

    joblist = []
    for sub_idx, sub in enumerate(SUBJECTS):
        dat = allsubj_data[sub_idx]
        other_subject_idx = np.setdiff1d(np.arange(len(SUBJECTS)), sub_idx)
        other_subs = np.setdiff1d(SUBJECTS, sub)
        other_subj_boundaries = [boundary_TRs_formatted[i] for i in other_subject_idx]
        joblist.append(delayed(run_single_subject)(dat, other_subj_boundaries, sub))

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

    demo = False
    if len(sys.argv) > 4:
        demo=True
        DATASET = "sherlock"
        ROI = "early_visual"
        METHOD="TPHATE"
        base_dir = utils.demo_dir
        data_dir = base_dir+'/demo_embeddings'
        outdir = '../results/demo_results'
        searchstr = 'sherlock_movie'
        param_df = pd.read_csv(f"{outdir}/within_sub_neural_event_WB_tempBalance.csv",index_col=0)
    else:
        SUBJECTS = utils.sherlock_subjects if DATASET == 'sherlock' else utils.forrest_subjects
        NTPTS = utils.sherlock_timepoints if DATASET == 'sherlock' else utils.forrest_movie_timepoints
        LOADFN = utils.load_sherlock_movie_ROI_data if DATASET == 'sherlock' else utils.load_forrest_movie_ROI_data
        base_dir = utils.sherlock_dir if DATASET == 'sherlock' else utils.forrest_dir
        data_dir = os.path.join(base_dir, utils.data_version, 'ROI_data', ROI)
        searchstr = 'sherlock_movie' if DATASET == 'sherlock' else 'forrest_movie'
        data_dir = os.path.join(data_dir, 'data') if METHOD == "voxel" else os.path.join(data_dir, 'embeddings')
        outdir='../results/'
        param_df = pd.read_csv(f"{outdir}/within_sub_neural_event_WB_tempBalance.csv",index_col=0)

    print(DATASET, ROI, METHOD, data_dir)

    NJOBS = 16
    param_df = param_df[(param_df['dataset'] == DATASET) & (param_df['ROI'] == ROI)
                        & (param_df['embed_method'] == METHOD)]
    SUBJECTS = param_df['subject'].values
    outfn = f'{outdir}/source/{ROI}_{DATASET}_{METHOD}_between_sub_neural_event_WB_tempBalance.csv'
    main()

