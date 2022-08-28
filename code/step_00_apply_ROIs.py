import numpy as np
import pandas as pd
import os, sys, glob
import nibabel as nib
import nilearn
import utils
from scipy.stats import zscore

def mask_sherlock(ROI_name):
    indata_dir = os.path.join(utils.sherlock_dir, utils.data_version, 'whole_brain_data')
    outdata_dir = os.path.join(utils.sherlock_dir, utils.data_version, 'ROI_data', ROI_name, 'data')
    os.makedirs(outdata_dir,exist_ok=True)
    ROI_mask = load_ROI(ROI_name)
    for subject in utils.sherlock_subjects:
        data = nib.load(f'{indata_dir}/sherlock_movie_s{subject}.nii')
        data_masked = utils.apply_volume_ROI(ROI_mask, data, z=True, zaxis=0) # zscore each voxel
        # check to make sure data is in the right shape
        if data_masked.shape[0] != utils.sherlock_timepoints:
            data_masked=data_masked.T
        outfn = f'{outdata_dir}/sub-{subject:02d}_{ROI_name}_sherlock_movie.npy'
        print(f'data of shape {data_masked.shape} going to {outfn}')
        np.save(outfn, data_masked)

def load_ROI(ROI_name):
    try:
        ROI_fn = os.path.join(utils.ROI_dir, f'{ROI_name}.nii.gz')
        ROI_mask = nib.load(ROI_fn)
    except:
        ROI_fn = os.path.join(utils.ROI_dir, f'{ROI_name}.nii')
        ROI_mask = nib.load(ROI_fn)
    return ROI_mask

# go through and combine the different runs into one dataset
def mask_forrest(ROI_name):
    indata_dir = os.path.join(utils.forrest_dir, 'whole_brain_data')
    outdata_dir = os.path.join(utils.forrest_dir, 'ROI_data', ROI_name, 'data')
    os.makedirs(outdata_dir, exist_ok=True)
    ROI_mask = load_ROI(ROI_name)
    for subject in utils.forrest_subjects:
        movie_data = []
        fns = sorted(glob.glob(f'{indata_dir}/sub-{subject:02d}*movie*')) # loop through all the runs and process separately
        for f in fns:
            data = nib.load(f)
            data_masked = utils.apply_volume_ROI(ROI_mask, data, outfn=False, z=True, zaxis=0)
            movie_data.append(data_masked)
        movie_data = np.concatenate(movie_data, axis=1)
        outfn = f'{outdata_dir}/sub-{subject:02d}_{ROI_name}_movie_all_runs.npy'
        if movie_data.shape[0] != utils.forrest_movie_timepoints:
            movie_data=movie_data.T
        print(outfn, movie_data.shape)
        np.save(outfn, movie_data)

        # now handle the localizer data
        loc_data = []
        fns = sorted(glob.glob(f'{indata_dir}/sub-{subject:02d}*objectcategories*'))
        for f in fns:
            data = nib.load(f)
            data_masked = utils.apply_volume_ROI(ROI_mask , data, outfn=False, z=True, zaxis=0)
            loc_data.append(data_masked)
        loc_data = np.concatenate(loc_data, axis=1)
        outfn = f'{outdata_dir}/sub-{subject:02d}_{ROI_name}_localizer_all_runs.npy'
        if loc_data.shape[0] != utils.forrest_localizer_timepoints:
            loc_data=loc_data.T
        print(outfn, loc_data.shape)
        np.save(outfn, loc_data)




if __name__ == '__main__':
    # handle both datasets and all ROIs
    ROIs = ['early_visual', 'high_Visual', 'pmc_nn', 'aud_early']
    for ROI in ROIs:
        mask_sherlock(ROI)
        print(f'finished sherlock {ROI}')
        mask_forrest(ROI)
        print(f'finished forrest {ROI}')

    # create embeddings














