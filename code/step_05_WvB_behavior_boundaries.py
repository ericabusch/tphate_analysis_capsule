# step_05_WvB_behavior_boundaries.py
"""
This script is to run the analysis presented in figure 7.
In this analysis, event boundaries identified by a separate cohort of human raters are applied
to the neural data at both voxel resolution and embedded with different methods.
These are
The fit of these boundaries are measured as in previous analyses (steps 3, 4, and 6).
This analysis can only be run for the Sherlock dataset, as the Forrest dataset doesn't have the
human rater labels.

To run as a demo, insert any second command-line argument

Runs from the command line as python step_05_WvB_behavior_boundaries.py $ROI $DEMO

Saves a csv file for the region to results directory.
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
from step_03_HMM_optimizeM_embeddings import compute_event_boundaries_diff_temporally_balanced

# this can only be performed for the sherlock data
def get_embedding_data(method, M_list):
    data = []
    for s,M in zip(utils.sherlock_subjects, M_list):
        fn = glob.glob(f'{embed_dir}/sub-{s:02d}*sherlock_movie*_{M}dimension_embedding_{method}.npy')[0]
        data.append(np.load(fn))
    return data

def get_scene_boundaries():
    sherlock_scenes_labels = utils.load_coded_sherlock_regressors('SceneTitleCoded')
    sherlock_scene_boundaries = [1] + list(np.where(np.diff(sherlock_scenes_labels))[0]) + [len(sherlock_scenes_labels)]
    return sherlock_scene_boundaries

if __name__ == "__main__":
    ROI = sys.argv[1]
    demo=True

    sherlock_scene_boundaries = get_scene_boundaries()
    print("Got boundaries")

    if len(sys.argv) > 2:
        demo=True
        print("Running demo; will adjust parameters")
        info_df = pd.read_csv('../results/demo_results/within_sub_neural_event_WB_tempBalance.csv', index_col=0)
        embed_dir = f'{utils.demo_dir}/demo_embeddings/'
        out_fn = f'../results/demo_results/source/{ROI}_demo_behavior_event_boundaries_WB_tempBalance.csv'
        METHODS=['TPHATE']
    else:
        info_df = pd.read_csv('../results/within_sub_neural_event_WB_tempBalance.csv', index_col=0)
        embed_dir = f'{utils.sherlock_dir}/ROI_data/{ROI}/embeddings/'
        out_fn = f'../results/source/{ROI}_behavior_event_boundaries_WB_tempBalance.csv'
        METHODS=utils.embedding_methods+['voxel']

    # read in to get the number of dimensions for embedding
    info_df = info_df[(info_df['dataset'] == 'sherlock') & (info_df['ROI'] == ROI)]
    results = pd.DataFrame(columns=['subject','embed_method','CV_M','avg_within','avg_between','avg_difference','ROI'])


    for METHOD in METHODS:
        info_df_here = info_df[(info_df['embed_method'] == METHOD)]
        M_list = info_df_here.sort_values(by='subject',axis=0)['CV_M'].values
        if METHOD == 'voxel':
            data = utils.load_sherlock_movie_ROI_data(ROI)
        else:
            data = get_embedding_data(METHOD, M_list)

        for subject, ds in zip(utils.sherlock_subjects, data):
            d, w, b, comparisons = compute_event_boundaries_diff_temporally_balanced(ds, sherlock_scene_boundaries)
            avg_within = np.nanmean(w)
            avg_between = np.nanmean(b)
            avg_diff = np.nanmean(d)
            results.loc[len(results)] = {'subject': subject,
                                         'embed_method': METHOD,
                                         'CV_M': ds.shape[1],
                                         'avg_within': avg_within,
                                         'avg_between': avg_between,
                                         'avg_difference': avg_diff,
                                         'ROI': ROI}
    results.to_csv(out_fn)

