# step_08_SVC_localizer.py
"""
Runs support vector classification on the object category localizer data from the StudyForrest dataset.
Performs train/test splits across runs, within-subject, for the two visual ROIs.
Uses a constant M=20 dimensionality for all embedding data. Reports classifier accuracy averaged across folds, 
since there's no temporal regularity to the stimuli.

Results presented in Figure 2, Supplementary Figure 3.

Runs from the command line as:
python step_08_SVC_localizer.py $METHOD
"""

import numpy as np
import pandas as pd
import os, sys, glob,random
import utils, config
import embedding_helpers as mf
from scipy.stats import zscore
from joblib import Parallel, delayed
from sklearn.svm import SVC
from sklearn.model_selection import PredefinedSplit


def get_embeddings(voxel_data_allsubj):
    embeddings = np.zeros((voxel_data_allsubj.shape[0], voxel_data_allsubj.shape[1], DIM))
    for i, s, ds in zip(np.arange(len(SUBJECTS)), SUBJECTS, voxel_data_allsubj):
        embed_fn = f'{EMBED_DIR}/sub-{s:02d}_{ROI}_localizer_{DIM}dimension_embedding_{METHOD}.npy'
        embed = mf.return_subject_embedding(embed_fn, ds, DIM, METHOD)
        embeddings[i, :, :] = embed
    return embeddings


def run_classifier(X, y, run_ids):
    parameters = {'C': 10, 'kernel': 'rbf'}
    ps = PredefinedSplit(run_ids) # leave one run out
    scores = []
    for train_index, test_index in ps.split():
        # Split the data according to the predefined split.
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = SVC(C=parameters['C'], kernel=parameters['kernel'])
        # Fit the SVM.
        model.fit(X_train, y_train)

        # Calculate the accuracy on the held out run.
        score = model.score(X_test, y_test)
        scores.append(score)
    return np.array(scores)


def main(ROI):
    results_df = pd.DataFrame(
        columns=['subject', 'embedding_dimensions', 
                 'embed_method', 'clf_accuracy', 'clf_sd', 'ROI'])
    EMBED_DIR = f'{BASE_DIR}/ROI_data/{ROI}/embeddings'
    data = []
    for sub in SUBJECTS:
        fn = f'{EMBED_DIR}/sub-{sub:02d}_{ROI}_localizer_{DIM}dimension_embedding_{METHOD}.npy'
        data.append(np.load(fn))
    data = np.array(data)
    if config.VERBOSE: print(f'loaded data of shape {data.shape}')
    run_ids = np.repeat(np.arange(4), TRs_per_run)
    joblist = []
    for i in range(len(data)):
        joblist.append(delayed(run_classifier)(data[i], LABELS[i], run_ids))
    with Parallel(n_jobs=NJOBS) as parallel:
        results = parallel(joblist)
    for i in range(len(results)):
        results_df.loc[len(results_df)] = {'subject': SUBJECTS[i],
                                           'embedding_dimensions': DIM,
                                           'ROI': ROI,
                                           'embed_method': METHOD,
                                           'clf_accuracy': np.nanmean(results[i]),
                                           'clf_sd': np.nanstd(results[i])}
    return results_df



if __name__ == '__main__':
    METHOD = sys.argv[1]
    DATASET='forrest'
    NJOBS = 16
    LABELS = utils.load_forrest_localizer_labels()  # list of one set of labels per subject
    SUBJECTS = config.SUBJECTS['forrest']
    NTPTS = config.LOCALIZER_TIMEPOINTS
    LOADFN = utils.load_forrest_localizer_ROI_data
    BASE_DIR = config.DATA_FOLDERS['forrest']
    OUT_DIR = config.RESULTS_FOLDERS[DATASET]
    searchstr = 'localizer'
    n_folds = 5
    ROIs = ['early_visual', 'high_Visual']
    DIM = 20 if METHOD != "TSNE" else 3
    TRs_per_run = 156

    for ROI in ROIs:
        print(f"Running {ROI} {DIM} {METHOD}")
        outfn = f'{OUT_DIR}/source/forrest_{ROI}_{METHOD}_SVC_localizer_results.csv'
        res = main(ROI)
        res.to_csv(outfn)
        print(f'saved {outfn}')









