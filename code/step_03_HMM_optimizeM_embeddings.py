# step_03_HMM_optimizeM_embeddings.py
"""

This script runs the analyses for figures 4 and 5 for the manifold embedding data.
It takes, for each subject, the cross-validated number of neural events identified by
step_02, and performs a similar cross-validation procedure to learn the M parameter -- the
 number of dimensions for each manifold that optiized the Within-vs-between event boundary difference.
Using the final CV_K and CV_M, for each subject, a new HMM is fit and the within-vs-between event boundary
difference is calculated, balancing the temporal distance between timepoints to assure fair comparison.
Saves this output, along with the log-likelihood of best model fit, to a csv file for each region, method, dataset.

Runs from the command line as:
python step_03_HMM_optimizeM_embeddings.py $DATASET $ROI $METHOD $DEMO
where demo is any additional argument, but runs the demo version.

"""


import numpy as np
import pandas as pd
import utils
import os, sys, glob
import brainiak.eventseg.event
from scipy.stats import zscore
from sklearn.model_selection import LeaveOneOut, KFold
from joblib import Parallel, delayed
import embedding_helpers as mf
from scipy.spatial.distance import pdist, cdist, squareform


def expand_boundary_labels(boundary_TRs, total_TRs):
    tp_event_TR_labels = []
    event = 0
    for i in range(1, len(boundary_TRs)):
        a = np.repeat(event, boundary_TRs[i] - boundary_TRs[i - 1])
        tp_event_TR_labels += list(a)
        event += 1
    for i in range(len(tp_event_TR_labels), total_TRs):
        tp_event_TR_labels.append(event + 1)
    return tp_event_TR_labels


def WvB_evalation(timeseries_data, boundary_TRs, buffer_size=4):
    # create a mask to buffer out 4 trs
    tpts = timeseries_data.shape[0]
    test_corrmat = 1 - squareform(pdist(timeseries_data, 'correlation'))
    mask_arr = np.ones((tpts, tpts))
    for i in range(tpts):
        for j in range(tpts):
            if np.abs(i - j) <= buffer_size:
                mask_arr[i][j] = np.nan
    corrmat_masked = test_corrmat * mask_arr
    W, B = [], []
    boundary_labels = expand_boundary_labels(boundary_TRs, tpts)
    for t0 in range(tpts):
        for t1 in range(t0, tpts):
            r = corrmat_masked[t0][t1]
            if boundary_labels[t0] == boundary_labels[t1]:
                W.append(r)
            else:
                B.append(r)
    return np.nanmean(W), np.nanmean(B)


def compute_event_boundaries_diff_temporally_balanced(timeseries_data, boundary_TRs):
    # make sure the boundaries inclue the start and end
    total_TRs = timeseries_data.shape[0]

    if not boundary_TRs[0] == 0:
        boundary_TRs = [0] + list(boundary_TRs)
    if not boundary_TRs[-1] == total_TRs:
        boundary_TRs += [total_TRs]

    timepoint_corrmat = 1 - squareform(pdist(timeseries_data, 'correlation'))
    # how long is the longest event?
    longest_event = np.max(np.diff(boundary_TRs))
    max_distance = longest_event
    print('Longest event ', max_distance)
    boundary_labels = expand_boundary_labels(boundary_TRs, total_TRs)
    # anchor on each timepoint
    comparisons_made = []
    between_event_correlations, within_event_correlations, diffs = [], [], []
    for anchor in np.arange(total_TRs):
        for distance in np.arange(1, max_distance + 1):
            # compare dist step forard and dist step back
            backward_tpt, forward_tpt = anchor - distance, anchor + distance
            # skip this if it doesnt fall in the bounds
            if backward_tpt <= 0 or forward_tpt >= total_TRs:
                continue
            # make sure one is within-event & one is between-event for the anchor.
            back_event, forward_event, anchor_event = boundary_labels[backward_tpt], boundary_labels[forward_tpt], \
                                                      boundary_labels[anchor]
            if check_events(back_event, anchor_event, forward_event, backward_tpt, forward_tpt, comparisons_made) == -1:
                continue
            # if we've made it this far, we can compare the two timepoint pairs & record that they were compared
            backward_corr = timepoint_corrmat[backward_tpt, anchor]
            forward_corr = timepoint_corrmat[forward_tpt, anchor]
            comparisons_made.append((backward_tpt, forward_tpt))
            # check which is between/within comparison
            if back_event == anchor_event:
                between_event_correlations.append(forward_corr)
                within_event_correlations.append(backward_corr)
                diffs.append(backward_corr - forward_corr)
            else:
                between_event_correlations.append(backward_corr)
                within_event_correlations.append(forward_corr)
                diffs.append(forward_corr - backward_corr)
    return diffs, within_event_correlations, between_event_correlations, comparisons_made


def test_boundaries_corr_diff(data, K, subject=None, balance_distance=False):
    total_TRs, n_features = data.shape
    HMM = brainiak.eventseg.event.EventSegment(K)
    HMM.fit(data)
    _, ll = HMM.find_events(data)

    boundary_TRs = np.where(np.diff(np.argmax(HMM.segments_[0], axis=1)))[0]
    try:
        if not boundary_TRs[0] == 0:
            boundary_TRs = [0] + list(boundary_TRs) + [total_TRs]
    except:
        print(f"===== broke on {subject} k={K} data of shape {data.shape}")
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    if balance_distance:
        d, w, b, comparisons = compute_event_boundaries_diff_temporally_balanced(data, boundary_TRs)
        avg_within = np.nanmean(w)
        avg_between = np.nanmean(b)
        avg_diff = np.nanmean(d)
    else:
        avg_within, avg_between = WvB_evalation(data, boundary_TRs)
        avg_diff = np.nan_to_num(avg_within - avg_between)
        comparisons = None
    return avg_within, avg_between, avg_diff, boundary_TRs, ll, comparisons


def check_events(backward_event, anchor_event, forward_event, backward_timepoint, forward_timepoint, comparisons_made):
    VALID = -1
    # first, check if these two timepoints were already compared
    if [backward_timepoint, forward_timepoint] in comparisons_made:
        print(f'{backward_timepoint}, {forward_timepoint} already compared')
        return VALID

    if backward_event == anchor_event:
        VALID *= -1
    if forward_event == anchor_event:
        VALID *= -1
    return VALID


def main():
    global SUBJECTS
    learnM_df = pd.DataFrame(columns=['avg_within_event_corr', 'avg_between_event_corr', 'avg_difference',
                               'M', 'test_subject', 'dataset', 'K', 'embed_method', 'ROI', 'boundary_TRs',
                               'model_LogLikelihood'])
    data = LOADFN(ROI)
    learnM_fn = f"{output_dir}/{ROI}_{DATASET}_{METHOD}_learnM.csv"
    if os.path.exists(learnM_fn):
        print(f"Loading {learnM_fn}")
        learnM_df=pd.read_csv(learnM_fn)
    else:
        ## drive paralel analysis
        joblist, parameters = [], []
        for i, subject in enumerate(SUBJECTS):
            bestK_cv = int(K_by_subject[i])
            test_data = data[i]
            for M in M2Test:
                embed_fn = f'{embed_dir}/sub-{subject:02d}_{ROI}_{DATASET}_movie_{M}dimension_embedding_{METHOD}.npy'
                embedding = mf.return_subject_embedding(embed_fn, test_data, M, METHOD)
                joblist.append(delayed(test_boundaries_corr_diff)(embedding, bestK_cv, subject=subject))
                parameters.append([M, subject, DATASET, bestK_cv, METHOD, ROI])
        with Parallel(n_jobs=NJOBS) as parallel:
            results = parallel(joblist)
        for r, p in zip(results, parameters):
            learnM_df.loc[len(learnM_df)] = {'avg_within_event_corr': r[0],
                               'avg_between_event_corr': r[1],
                               'avg_difference': r[2],
                               'boundary_TRs': r[3],
                               'model_LogLikelihood': r[4],
                               'M': p[0],
                               'test_subject': p[1],
                               'dataset': p[2],
                               'K': p[3],
                               'embed_method': p[4],
                               'ROI': p[5]}
        # this is where we cross-validate the dimensionality
        learnM_df.to_csv(learnM_fn)
        print(f"saved to {learnM_fn}")

    # now we have the dataframe with the best M's - find the best by subject
    bestM_within_subject = []
    incl = []
    SUBJECTS = utils.sherlock_subjects if DATASET == 'sherlock' else utils.forrest_subjects
    for j, subject in enumerate(SUBJECTS):
        sub_df = learnM_df[learnM_df['test_subject'] == subject]
        max_diff = np.max(sub_df['avg_difference'].values)
        bestM = sub_df[sub_df['avg_difference'] == max_diff]['M'].values[0]
        incl.append(subject)
        bestM_within_subject.append(bestM)
        # # get the highest difference value
        # try:
        #     max_diff = np.max(sub_df['avg_difference'].values)
        #     bestM = sub_df[sub_df['avg_difference'] == max_diff]['M'].values[0]
        #     incl.append(subject)
        #     bestM_within_subject.append(bestM)
        # except:
        #     print(f"excluding {subject}")
        #     continue
    SUBJECTS = incl

    final_df = pd.DataFrame(columns=['subject', 'ROI', 'dataset', 'avg_within_event_corr', 'avg_between_event_corr', 'avg_difference',
                 'CV_M', 'CV_K', 'embed_method', 'boundary_TRs', 'model_LogLikelihood', 'compared_timepoints'])

    # now cross validate - each subject gets the mean of all the others
    for i, subject in enumerate(SUBJECTS):
        test_data = data[i]
        bestK_cv = int(K_by_subject[i])
        others = np.setdiff1d(np.arange(len(SUBJECTS)), i)
        bestM_cv = int(np.round(np.nanmean([bestM_within_subject[j] for j in others])))
        print(
            f'bestM_cv for subject {subject} method {METHOD} dataset {DATASET} roi {ROI} is {bestM_cv}; bestK {bestK_cv}')
        embed_fn = f'{embed_dir}/sub-{subject:02d}_{ROI}_{DATASET}_movie_{bestM_cv}dimension_embedding_{METHOD}.npy'
        embedding = mf.return_subject_embedding(embed_fn, test_data, bestM_cv, METHOD)

        avg_within, avg_between, avg_diff, boundary_TRs, ll, comparisons = test_boundaries_corr_diff(embedding, bestK_cv, balance_distance=True)
        final_df.loc[len(final_df)] = {'subject': subject,
                                       'ROI': ROI,
                                       'dataset': DATASET,
                                       'avg_within_event_corr': avg_within,
                                       'avg_between_event_corr': avg_between,
                                       'avg_difference': avg_diff,
                                       'CV_M': bestM_cv,
                                       'CV_K': bestK_cv,
                                       'embed_method': METHOD,
                                       'boundary_TRs': boundary_TRs,
                                       'model_LogLikelihood': ll,
                                       'compared_timepoints': comparisons}
    final_df.to_csv(outfn_name)
    print(f'saved {outfn_name}')


if __name__ == "__main__":
    DATASET = sys.argv[1]
    ROI = sys.argv[2]
    METHOD = sys.argv[3]
    NJOBS=16
    if len(sys.argv) > 4:
        print("Running demo; adjusting parameters accordingly")
        ROI='early_visual'
        METHOD='TPHATE'
        DATASET='sherlock'
        SUBJECTS=utils.sherlock_subjects
        NTPTS=utils.sherlock_timepoints
        LOADFN=utils.load_sherlock_movie_ROI_data
        base_dir = "../data/demo_data/"
        data_dir = f'{base_dir}/demo_ROI_data/'
        LOADFN = utils.load_demo_data
        embed_dir = f'{base_dir}/demo_embeddings/'
        output_dir='../intermediate_data/demo/HMM_learnK'
        results_dir = '../results/demo_results/source'
    else:
        SUBJECTS = utils.sherlock_subjects if DATASET == 'sherlock' else utils.forrest_subjects
        NTPTS = utils.sherlock_timepoints if DATASET == 'sherlock' else utils.forrest_movie_timepoints
        LOADFN = utils.load_sherlock_movie_ROI_data if DATASET == 'sherlock' else utils.load_forrest_movie_ROI_data
        SUBJECTS = utils.sherlock_subjects if DATASET == 'sherlock' else utils.forrest_subjects
        NTPTS = utils.sherlock_timepoints if DATASET == 'sherlock' else utils.forrest_movie_timepoints
        LOADFN = utils.load_sherlock_movie_ROI_data if DATASET == 'sherlock' else utils.load_forrest_movie_ROI_data
        base_dir = utils.sherlock_dir if DATASET == 'sherlock' else utils.forrest_dir
        embed_dir = f'{base_dir}/ROI_data/{ROI}/embeddings/'
        output_dir = '../intermediate_data/HMM_learnK'
        results_dir = '../results/source'

    bestK_df_fn = f'{output_dir}/{DATASET}_{ROI}_bestK_LOSO.csv'
    outfn_name = f'{results_dir}/{ROI}_{DATASET}_{METHOD}_tempBalance_crossValid_WB_results.csv'
    bestK_df = pd.read_csv(bestK_df_fn, index_col=0)  # this has the best K when holding out each subject
    bestK_df = bestK_df[(bestK_df['ROI'] == ROI)]
    K_by_subject = bestK_df['CV_K'].values
    SUBJECTS=list(SUBJECTS)
    M2Test = np.arange(2, 11, 1) if METHOD != 'TSNE' else np.arange(2, 4) # tsne can only embed in 2 or 3
    main()

