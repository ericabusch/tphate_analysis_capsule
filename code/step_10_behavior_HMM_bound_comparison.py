import numpy as np
import pandas as pd
import utils, config
import brainiak.eventseg.event.EventSegment
import scipy.stats
from scipy.stats import norm, zscore, pearsonr

def run_permutations(human_bounds, bounds, buffer):
    if bounds[0]!=0:
        bounds=np.concatenate(([0], bounds, [nTR]))
    nPerm=1000
    event_counts = np.diff(bounds)
    match = np.zeros(nPerm + 1)
    perm_bounds = bounds
    true_matches = 0
    for p in range(nPerm + 1):
        for hb in human_bounds:
            if np.any(np.abs(perm_bounds - hb) <= buffer):
                match[p] += 1
                if p == 0:
                    true_matches += 1
        match[p] /= len(human_bounds)
        perm_counts = np.random.permutation(event_counts)
        perm_bounds = np.cumsum(perm_counts)[:-1]
    p = norm.sf((match[0]-match[1:].mean())/match[1:].std())
    z = scipy.stats.zscore(match)[0]
    return p, z, true_matches
    
def boundary_distance(human_bounds, bounds):
    if bounds[0]!=0:
        bounds=np.concatenate(([0], bounds, [nTR]))
    distances = np.zeros_like(human_bounds)
    for j,hb in enumerate(human_bounds):
        distances[j] = np.abs(hb - bounds).min()
    return np.mean(distances), np.median(distances)

if __name__ === "__main__":
    ROI = "pmc_nn"
    K = 31 # number of behavioral events
    nTR = 1976 # total TRs
    buffer = 10
    # load in file for optimized within-sub HMM results

    parm_fn = f'../results/within_sub_neural_event_WB_tempBalance_results.csv'
    param_df = pd.read_csv(parm_fn)[['ROI','CV_M','CV_K','dataset','subject','embed_method']]
    human_bounds = np.array(utils.get_scene_boundaries())
    buff_res = pd.DataFrame(columns=['subject','method', 'K', 'zscore', 
    'match','pval','buffer', 'min_dist_mean','min_dist_med'])
    for meth in ['UMAP','PCA','TPHATE','PHATE']:
        # grab embedding files
        for i,sub in enumerate(np.arange(1,17)):
            CV_M = int(param_df[(param_df['subject'] == i+1)&(param_df['embed_method'] == meth)]['CV_M'])
            file = sorted(glob.glob(dirname+f'/sub-{sub:02d}*{CV_M}dimension_embedding_{meth}.npy'))[0]
            seg_fn = f'../intermediate_data/sub-{i+1:02d}_HMM_segments_{meth}_{CV_M}dim.npy'
            if os.path.exists(seg_fn):
                segmentation = np.load(seg_fn)
            else:
                data = np.load(file)
                HMM = brainiak.eventseg.event.EventSegment(K)
                HMM.fit(data)
                np.save(seg_fn, HMM.segments_[0])
                segmentation = HMM.segments_[0]
            bounds = np.where(np.diff(np.argmax(segmentation, axis=1)))[0]
            p, z, true_matches = run_permutations(bounds, human_bounds,buffer)
            mean, med = boundary_distance(human_bounds, bounds)
            buff_res.loc[len(buff_res)] = {'subject':i+1, 'method':meth, 
                                           'K':31, 'zscore':z, 'pval':p, 
                                           'match':true_matches,
                                           'buffer':buffer,'min_dist_mean':mean,
                                           'min_dist_med': med}
    buff_res.to_csv("../results/human_bounds_behav_bounds_sameK_final_CV_M.csv")                                       

















