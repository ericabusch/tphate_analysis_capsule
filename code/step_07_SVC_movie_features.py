import numpy as np
import pandas as pd
import os, sys, glob, random
import utils
import embedding_helpers as mf
import scipy.stats
from joblib import Parallel, delayed
from sklearn.svm import SVC
from sklearn.model_selection import KFold


def random_upsample_train_data(X_train, Y_train):
    (unique, counts) = np.unique(Y_train, return_counts=True)
    goal = max(counts)

    train_indices = []
    for ind, classs in enumerate(unique):
        if counts[ind] != goal:
            num2add = goal - counts[ind]
            where_this_class = np.where(Y_train == classs)
            chosen_indices = list(random.choices(where_this_class[0], k=num2add)) + list(where_this_class[0])
            train_indices = train_indices + chosen_indices
        else:
            train_indices += list(np.where(Y_train == classs)[0])
    train_indices.sort()
    X_train = X_train[train_indices]
    Y_train = Y_train[train_indices]
    return X_train, Y_train


def shift_labels(labels, shift_by, rep_number):
    TRs_total = np.arange(labels.shape[0])
    TRs_to_shift = np.arange(0, shift_by * rep_number)
    TRs_to_keep = np.setdiff1d(TRs_total, TRs_to_shift)
    shifted_TRs = np.concatenate((TRs_to_keep, TRs_to_shift))
    try:
        shifted_labels = labels[shifted_TRs]
    except:
        shifted_labels = np.array([labels[i] for i in shifted_TRs if i < len(labels)])
    return shifted_labels


def load_embeddings(voxel_data_allsubj):
    embeddings = np.zeros((voxel_data_allsubj.shape[0], voxel_data_allsubj.shape[1], DIM))
    for i, s, ds in zip(np.arange(embeddings.shape[0]), SUBJECTS, voxel_data_allsubj):
        embed_fn = f'{embedding_dir}/sub-{s:02d}_{ROI}_{searchstr}_{DIM}dimension_embedding_{METHOD}.npy'
        embedding = mf.return_subject_embedding(embed_fn, ds, DIM, METHOD)
        embeddings[i, :, :] = embedding
    return embeddings


def shift_labels_and_run_clf(X, y, num_folds_per_shift, shift_by, num_reps):
    parameters = {'C': 10, 'kernel': 'rbf'}
    n_samples, n_features = X.shape
    results = np.zeros((num_reps, num_folds_per_shift))
    # for each repetition, then fold, then shift the training labels
    for rep in range(num_reps):
        folder = KFold(n_splits=num_folds_per_shift, shuffle=True)
        for fold, (train, test) in enumerate(folder.split(np.arange(n_samples))):
            # now that we are here, rolls the training labels according to the rep
            y_train = shift_labels(y[train], shift_by, rep)
            if np.sum(np.isnan(y_train)) == 1:  # need to make sure that not all the labels are the same
                results[rep, fold] = np.nan
                continue
            # now balance the training data to have equal exemplars from class
            # even though we've broken the structure between X and y
            X_train, y_train = random_upsample_train_data(X[train], y_train)
            svc = SVC(kernel=parameters['kernel'], C=parameters['C'])
            svc.fit(X_train, y_train)
            results[rep, fold] = svc.score(X[test], y[test])
    acc_per_roll = np.nanmean(results, axis=1)  # average across data folds
    zscored_acc = scipy.stats.zscore(acc_per_roll)  # zscore all the values
    true_score_zstat = zscored_acc[0]  # the first one is rep=0 (no shift)
    p_value = scipy.stats.norm.sf(true_score_zstat)
    return true_score_zstat, p_value


def main():
    data = np.array(LOADFN(ROI))

    if METHOD != 'voxel':
        data = load_embeddings(data)

    shift_by = int(data.shape[1] / num_reps)

    joblist, parameters = [], []
    for reg_name, labels in REGRESSORS.items():
        for sub_idx, sub in enumerate(SUBJECTS):
            joblist.append(
                delayed(shift_labels_and_run_clf)(data[sub_idx], labels, n_folds_per_shift, shift_by, num_reps))
            parameters.append((sub, DIM, METHOD, reg_name, ROI, DATASET, shift_by, num_reps))

    with Parallel(n_jobs=NJOBS) as parallel:
        results_list = parallel(joblist)

    df = pd.DataFrame(columns=["ROI", "dataset", 'subject', 'embed_method', 'regressor', 'shift_by', 'num_reps',
                               'embed_dimensions', 'zstat', 'p-value'])

    for r, p in zip(results_list, parameters):
        df.loc[len(df)] = {"subject": p[0],
                           "embed_dimensions": p[1],
                           "embed_method": p[2],
                           'regressor': p[3],
                           'ROI': p[4],
                           'dataset': p[5],
                           'shift_by': p[6],
                           'num_reps': p[7],
                           'zstat': r[0],
                           'p-value': r[1]
                           }
    df.to_csv(outfn)
    print(f"saved to {outfn}")


if __name__ == "__main__":
    DATASET = sys.argv[1]
    ROI = sys.argv[2]
    METHOD = sys.argv[3]
    NJOBS = 16
    DIM = 20 if METHOD != "TSNE" else 3
    n_folds_per_shift = 3
    num_reps = 1000
    demo = False

    if DATASET == "sherlock":
        REGRESSORS = {'IndoorOutdoor': utils.load_coded_sherlock_regressors('IndoorOutdoor'),
                      'MusicPresent': utils.load_coded_sherlock_regressors('MusicPresent')}
        SUBJECTS = utils.sherlock_subjects
        NTPTS = utils.sherlock_timepoints
        LOADFN = utils.load_sherlock_movie_ROI_data
        base_dir = utils.sherlock_dir
        searchstr = 'sherlock_movie'
    else:
        REGRESSORS = {'IoE_coded': utils.load_coded_forrest_movie_regressors('IoE_coded'),
                      'FoT_coded': utils.load_coded_forrest_movie_regressors("FoT_coded")}
        SUBJECTS = utils.forrest_subjects
        NTPTS = utils.forrest_movie_timepoints
        LOADFN = utils.load_forrest_movie_ROI_data
        base_dir = utils.forrest_dir
        searchstr = 'forrest_movie'

    if len(sys.argv) > 4:
        demo = True
        outdir = '../results/demo_results/source'
        base_dir = utils.demo_dir
        embedding_dir = f'{base_dir}/demo_embeddings/'
        LOADFN=utils.load_demo_data
    else:
        outdir = '../results/source'
        embedding_dir = f'{base_dir}/ROI_data/{ROI}/embeddings/'

    outfn = f'{outdir}/{DATASET}_{ROI}_{METHOD}_SVC_movie_zstat_results.csv'
    print(f"Running {ROI} {DATASET} {METHOD} | saving to {outfn}")
    main()

