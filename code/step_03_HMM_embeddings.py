# step_03_HMM_embeddings.py
"""

This script takes, for each subject, the cross-validated number of neural events identified by
step_02, and performs a similar cross-validation procedure to learn the M parameter -- the
 number of dimensions for each embedding that optimized the within-vs-between event boundary difference.
Using the final CV_K and CV_M, for each subject, a new HMM is fit and the within-vs-between event boundary
difference is calculated, balancing the temporal distance between timepoints to assure fair comparison. This is run only
for dimensionality-reduced methods, to learn the # dimensions to reduce each method to, allowing for distinctions
across methods.
Saves this output, along with the log-likelihood and AIC of best model fit, to a csv file for each region, method, dataset.
This script also runs the same analysis without the optimize M step -- instead using a constant 3 dimensional embedding -- and saves as a control dataset to test for the effect of dimensionality. 

Runs from the command line as:
python step_03_HMM_embeddings.py $DATASET $ROI $METHOD

"""


import numpy as np
import pandas as pd
import utils, config
from config import NJOBS
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

    tpts = timeseries_data.shape[0]
    test_corrmat = 1 - squareform(pdist(timeseries_data, 'correlation'))
    mask_arr = np.zeros((tpts, tpts))
    mask_arr[np.triu_indices_from(mask_arr, buffer_size)] = 1
    mask_arr+=mask_arr.T
    mask_arr[mask_arr==0]=np.nan


    corrmat_masked = test_corrmat * mask_arr
    W, B = [], []
    boundary_labels = expand_boundary_labels(boundary_TRs, tpts)
    # create a mask to buffer out `buffer_size` TRs

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
    max_distance = np.max(np.diff(boundary_TRs))
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
    return diffs, within_event_correlations, between_event_correlations, comparisons_made, boundary_labels


def test_boundaries_corr_diff(data, K, subject=None, balance_distance=False):
    total_TRs, n_features = data.shape
    HMM = brainiak.eventseg.event.EventSegment(K)
    HMM.fit(data)
    _, ll = HMM.find_events(data)
    
    AIC = -2 * ll + 2*(K**2 +2*K - 1) # AIC=−2logL+2p, p=m^2+km−1.
    
    boundary_TRs = np.where(np.diff(np.argmax(HMM.segments_[0], axis=1)))[0]
    
    if not boundary_TRs[0] == 0:
        boundary_TRs = [0] + list(boundary_TRs) + [total_TRs]

    if balance_distance:
        d, w, b, comparisons, event_labels = compute_event_boundaries_diff_temporally_balanced(data, boundary_TRs)
        avg_within = np.nanmean(w)
        avg_between = np.nanmean(b)
        avg_diff = np.nanmean(d)
    else:
        avg_within, avg_between = WvB_evalation(data, boundary_TRs)
        avg_diff = np.nan_to_num(avg_within - avg_between)
        comparisons = None
        event_labels = None
    return avg_within, avg_between, avg_diff, boundary_TRs, ll, AIC, comparisons, event_labels

# Check whether two timepoints are a valid comparison:
# One has to be within the same event as the anchor and one across events, they hve to be equidistant from the anchor TP
# and they have to be not already compared
def check_events(backward_event, anchor_event, forward_event, backward_timepoint, forward_timepoint, comparisons_made):
    VALID = -1
    # first, check if these two timepoints were already compared
    if [backward_timepoint, forward_timepoint] in comparisons_made:
        if config.VERBOSE: print(f'{backward_timepoint}, {forward_timepoint} already compared')
        return VALID
    if backward_event == anchor_event:
        VALID *= -1
    if forward_event == anchor_event:
        VALID *= -1
    return VALID


def main():
    learnM_df = pd.DataFrame(columns=['avg_within_event_corr', 'avg_between_event_corr', 'avg_difference',
                               'M', 'subject', 'dataset', 'K', 'embed_method', 'ROI', 'boundary_TRs',
                               'model_LogLikelihood'])
    D = 'sherlock' if DATASET == 'demo' else DATASET
    data = LOADFN(ROI)
    if os.path.exists(learnM_fn):
        if config.VERBOSE: print(f"Loading {learnM_fn}")
        learnM_df=pd.read_csv(learnM_fn) # if this has already been run, don't repeat
    else:
        ## drive paralel analysis
        joblist, parameters = [], []
        for i, subject in enumerate(SUBJECTS):
            bestK_cv = int(K_by_subject[i])
            test_data = data[i]
            for M in M2Test:
                embed_fn = f'{EMBED_DIR}/sub-{subject:02d}_{ROI}_{D}_movie_{M}dimension_embedding_{METHOD}.npy'
                embedding = mf.return_subject_embedding(embed_fn, test_data, M, METHOD)
                joblist.append(delayed(test_boundaries_corr_diff)(embedding, bestK_cv, subject=subject))
                parameters.append([M, subject, DATASET, bestK_cv, METHOD, ROI])
        with Parallel(n_jobs=NJOBS) as parallel:
            results = parallel(joblist)
        for r, p in zip(results, parameters):
            avg_within, avg_between, avg_diff, boundary_TRs, ll, AIC, _, _ = r
            
            learnM_df.loc[len(learnM_df)] = {'avg_within_event_corr': avg_within,
                               'avg_between_event_corr': avg_between,
                               'avg_difference': avg_diff,
                               'boundary_TRs': boundary_TRs,
                               'model_LogLikelihood': ll,
                               'AIC':AIC,             
                               'M': p[0],
                               'subject': p[1],
                               'dataset': p[2],
                               'K': p[3],
                               'embed_method': p[4],
                               'ROI': p[5]}
        # this is where we cross-validate the dimensionality
        learnM_df.to_csv(learnM_fn)
        if config.VERBOSE: print(f"saved to {learnM_fn}")

    # now we have the dataframe with scores for all M's - find cross-validated M per test subject
    CV_M_by_subject = []
    for j, test_subject in enumerate(SUBJECTS):
        training_subjects = np.setdiff1d(SUBJECTS, test_subject)
        sub_df = learnM_df[learnM_df['subject'].isin(training_subjects)] # exclude current subject
        avgd = sub_df.groupby(['M']).mean().reset_index() # average within values of M across training subjects
        # find the M that maximizes the average difference at the group level
        CV_M = int(avgd[avgd['avg_difference'] == avgd.avg_difference.max()]['M'])
        CV_M_by_subject.append(CV_M)
  
    final_df = pd.DataFrame(columns=['subject', 'ROI', 'dataset', 'avg_within_event_corr', 
                                     'avg_between_event_corr', 'avg_difference', 'CV_M', 
                                     'CV_K', 'embed_method', 'boundary_TRs', 'model_LogLikelihood', 'AIC',
                                     'compared_timepoints'])
    
    control_df = pd.DataFrame(columns=['subject','ROI','dataset','avg_within_event_corr', 
                                     'avg_between_event_corr', 'avg_difference','M','CV_K','embed_method',
                                       'boundary_TRs','model_LogLikelihood','AIC','compared_timepoints'])
    joblist = []
    parameters = []
    # now cross validate - each subject gets the mean of all the others
    for i, subject in enumerate(SUBJECTS):
        test_data = data[i]
        CV_K = int(K_by_subject[i])
        train_subject_inds = np.setdiff1d(np.arange(len(SUBJECTS)), i)
        
        # run the embedding with optimized M
        CV_M = int(CV_M_by_subject[i])
        if config.VERBOSE: print(f'Subject {subject} K={CV_K} M={CV_M}')
        embed_fn = f'{EMBED_DIR}/sub-{subject:02d}_{ROI}_{D}_movie_{CV_M}dimension_embedding_{METHOD}.npy'
        parameters.append({'subject':subject, 'M':CV_M, 'K':CV_K, 'optimizeM':True})
        embedding = mf.return_subject_embedding(embed_fn, test_data, CV_M, METHOD)
        joblist.append(delayed(test_boundaries_corr_diff)(embedding, CV_K, balance_distance=True))
        
        # run the embedding with constant M=3
        constant_M = 3
        embed_fn = f'{EMBED_DIR}/sub-{subject:02d}_{ROI}_{D}_movie_{constant_M}dimension_embedding_{METHOD}.npy'
        parameters.append({'subject':subject, 'M':constant_M, 'K':CV_K, 'optimizeM':False})
        embedding = mf.return_subject_embedding(embed_fn, test_data, constant_M, METHOD)
        joblist.append(delayed(test_boundaries_corr_diff)(embedding, CV_K, balance_distance=True))
    
    if config.VERBOSE: print('starting parallel final results')
    with Parallel(n_jobs=NJOBS) as parallel:
        results = parallel(joblist)
    if config.VERBOSE: print('ending parallel final results; saving results')
        
    for r, p in zip(results, parameters):
        avg_within, avg_between, avg_diff, boundary_TRs, ll, AIC, comparisons, event_labels = r
        subject = p['subject']
        CV_K = p['K']
        if p['optimizeM']:
            CV_M = p['M']
            # save boundaryTRs in intermediate data dir for subsequent analysis
            saveto = f"{OUT_DIR}/sub-{subject:02d}_{ROI}_{D}_movie_{CV_M}dimension_embedding_{METHOD}_HMM_boundaries.npy"
            np.save(saveto, boundary_TRs)
            # save compared TRs in intermediate data dir for subsequent analysis
            saveto = f"{OUT_DIR}/sub-{subject:02d}_{ROI}_{D}_movie_{CV_M}dimension_embedding_{METHOD}_WvB_comparisons.npy"
            np.save(saveto, comparisons)
        
            final_df.loc[len(final_df)] = {'subject': subject,
                                           'ROI': ROI,
                                           'dataset': DATASET,
                                           'avg_within_event_corr': avg_within,
                                           'avg_between_event_corr': avg_between,
                                           'avg_difference': avg_diff,
                                           'CV_M': CV_M,
                                           'CV_K': CV_K,
                                           'AIC':AIC,
                                           'embed_method': METHOD,
                                           'boundary_TRs': boundary_TRs,
                                           'model_LogLikelihood': ll,
                                           'compared_timepoints': comparisons}
        else:
            # save boundaryTRs in intermediate data dir for subsequent analysis
            saveto = f"{OUT_DIR}/sub-{subject:02d}_{ROI}_{D}_movie_control_embedding_{METHOD}_HMM_boundaries.npy"
            np.save(saveto, boundary_TRs)
            # save event labels in intermediate data dir for subsequent analysis
            saveto = f"{OUT_DIR}/sub-{subject:02d}_{ROI}_{D}_movie_control_embedding_{METHOD}_HMM_boundaries.npy"
            np.save(saveto, boundary_TRs)
            # save compared TRs in intermediate data dir for subsequent analysis
            saveto = f"{OUT_DIR}/sub-{subject:02d}_{ROI}_{D}_movie_control_embedding_{METHOD}_WvB_comparisons.npy"
            np.save(saveto, comparisons)
        
            control_df.loc[len(control_df)] = {'subject': subject,
                                           'ROI': ROI,
                                           'dataset': DATASET,
                                           'avg_within_event_corr': avg_within,
                                           'avg_between_event_corr': avg_between,
                                           'avg_difference': avg_diff,
                                           'CV_M': CV_M,
                                           'CV_K': CV_K,
                                           'AIC':AIC,
                                           'embed_method': METHOD,
                                           'boundary_TRs': boundary_TRs,
                                           'model_LogLikelihood': ll,
                                           'compared_timepoints': comparisons}
                        
    final_df.to_csv(outfn_name)
    control_df.to_csv(control_name)
    if config.VERBOSE: print(f'saved {outfn_name} and {control_name}')


if __name__ == "__main__":
    DATASET = sys.argv[1]
    ROI = sys.argv[2]
    METHOD = sys.argv[3]

    SUBJECTS = config.SUBJECTS[DATASET]
    NTPTS = config.TIMEPOINTS[DATASET]
    LOADFN = utils.LOAD_FMRI_FUNCTIONS[DATASET]
    BASE_DIR = config.DATA_FOLDERS_CAPSULE[DATASET]
    DATA_DIR = f'{BASE_DIR}/demo_ROI_data' if DATASET == 'demo' else f'{BASE_DIR}/ROI_data/{ROI}/data'
    EMBED_DIR = f'{BASE_DIR}/demo_embeddings' if DATASET == 'demo' else f'{BASE_DIR}/ROI_data/{ROI}/embeddings'
    M2Test = config.DIMENSIONS_TO_TEST if METHOD != 'TSNE' else [2,3]
    
    OUT_DIR = config.INTERMEDIATE_DATA_FOLDERS[DATASET] + '/HMM_learnK_nested'
    RESULTS_DIR =config.RESULTS_FOLDERS[DATASET]+'/source'    
    bestK_df_fn = f'{OUT_DIR}/{DATASET}_{ROI}_bestK_nestedCV.csv'
    learnM_fn = f"{OUT_DIR}/{ROI}_{DATASET}_{METHOD}_learnM.csv"
    outfn_name = f'{RESULTS_DIR}/{ROI}_{DATASET}_{METHOD}_tempBalance_crossValid_WB_results.csv'
    control_name = f'{RESULTS_DIR}/{ROI}_{DATASET}_{METHOD}_tempBalance_control_WB_results.csv'
    
    if config.VERBOSE: print(f"loaded {bestK_df_fn}; results to {outfn_name}")
        
    bestK_df = pd.read_csv(bestK_df_fn, index_col=0)  # this has the best K when holding out each subject
    bestK_df = bestK_df[(bestK_df['ROI'] == ROI)]
    K_by_subject = bestK_df['CV_K'].values
    main()

