import numpy as np
import pandas as pd
import os, sys, glob,random
import utils
import embedding_helpers as mf
from scipy.stats import zscore
from joblib import Parallel, delayed
from sklearn.svm import SVC
from sklearn.model_selection import PredefinedSplit


def get_embeddings(voxel_data_allsubj, embed_dir):
    embeddings = np.zeros((voxel_data_allsubj.shape[0], voxel_data_allsubj.shape[1], DIM))
    for i, s, ds in zip(np.arange(len(SUBJECTS)), SUBJECTS, voxel_data_allsubj):
        embed_fn = f'{embed_dir}/sub-{s:02d}_{ROI}_localizer_{DIM}dimension_embedding_{METHOD}.npy'
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
        columns=['subject', 'embedding_dimensions', 'embed_method', 'clf_accuracy', 'clf_sd', 'ROI'])
    data = np.array(LOADFN(ROI))
    if METHOD != 'voxel':
        data = get_embeddings(data, embed_dir)
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
    NJOBS = 16
    LABELS = utils.load_forrest_localizer_labels()  # list of one set of labels per subject
    SUBJECTS = utils.forrest_subjects
    NTPTS = utils.forrest_movie_timepoints
    LOADFN = utils.load_forrest_localizer_ROI_data
    base_dir = utils.forrest_dir
    searchstr = 'localizer'
    n_folds = 5
    ROIs = ['early_visual', 'high_Visual']
    DIM = 20 if METHOD != "TSNE" else 3
    TRs_per_run = 156

    for ROI in ROIs:
        print(f"Running {ROI} {DIM} {METHOD}")
        ROI_dir = os.path.join(base_dir, utils.data_version, 'ROI_data', ROI)
        embed_dir = f'{utils.forrest_dir}/ROI_data/{ROI}/embeddings'
        outfn = f'../results/source/forrest_{ROI}_{METHOD}_SVC_localizer_results.csv'
        r = main(ROI)
        r.to_csv(outfn)
        print(f'saved {outfn}')









