# analyze significance of the match between HMM identified boundaries and 
# behavioral boundaries 
import numpy as np
import pandas as pd
import os,sys,glob
import scipy.stats
from scipy.stats import norm, zscore, pearsonr
import config, utils

np.random.seed(0)
nPerm=1000
nTR=1976
human_bounds = utils.get_scene_boundaries()

results_df = pd.DataFrame(columns=['ROI','subject','embed_method','p_value','match','zscore','K_HMM'])

#this is where we saved the HMM identified bounds
bdir = f'../intermediate_data/HMM_learnK_nested/'  

for method in config.EMBEDDING_METHODS+['voxel']:
    for roi in config.ROIs:
        for i, sub in enumerate(config.SUBJECTS['sherlock']):
            fn=glob.glob(f'{bdir}/sub-{sub:02d}*{roi}*sherlock*{method}*HMM_bound*')[0]
            bounds = np.load(fn)
            if bounds[0]!=0:
                bounds=np.concatenate(([0], bounds, [nTR]))
            event_counts = np.diff(bounds)
            match = np.zeros(nPerm+1)
            perm_bounds = bounds
            for p in range(nPerm+1):
                for hb in human_bounds:
                    if np.any(np.abs(perm_bounds - hb) <= 3):
                        match[p] += 1
                match[p] /= len(human_bounds)

                perm_counts = np.random.permutation(event_counts)
                perm_bounds = np.cumsum(perm_counts)[:-1]

            p = norm.sf((match[0]-match[1:].mean())/match[1:].std())
            z = scipy.stats.zscore(match)[0]
            results_df.loc[len(results_df)] = {'ROI':roi, 
            'subject':sub, 
            'embed_method':method, 
            'p_value':p,
            'match':match[0],
            'K_HMM':len(bounds)-1,
            'zscore':z}
        print(f'done {method} {roi}')
results_df.to_csv("../results/hmm_behav_bound_match.csv")

