## TO DO: UPDATE TO NEW FILE STRUCTURE

import numpy as np
import pandas as pd
import os, sys, glob
import nibabel as nib
import nilearn
import utils
from scipy.stats import zscore
from config import *

def load_ROI(ROI_name):
    ROI_mask = nib.load(config.ROI_FILES[ROI_name])
    return ROI_mask

def mask_sherlock(ROI_name):
    indata_dir = RAW_DATA_FOLDERS['sherlock']
    outdata_dir = DATA_FOLDERS['sherlock']
    os.makedirs(outdata_dir, exist_ok=True)
    ROI_mask = load_ROI(ROI_name)
    for subject in SUBJECTS['sherlock']:
        data = nib.load(f'{indata_dir}/sherlock_movie_s{subject}.nii')
        data_masked = utils.apply_volume_ROI(ROI_mask, data, z=True, zaxis=0) # zscore each voxel
        # check to make sure data is in the right shape
        if data_masked.shape[0] != TIMEPOINTS['sherlock']:
            data_masked=data_masked.T
        outfn = f'{outdata_dir}/sub-{subject:02d}_{ROI_name}_sherlock_movie.npy'
        print(f'data of shape {data_masked.shape} going to {outfn}')
        np.save(outfn, data_masked)

def mask_forrest(ROI_name):
    indata_dir = RAW_DATA_FOLDERS['forrest']
    outdata_dir = DATA_FOLDERS['forrest']
    os.makedirs(outdata_dir, exist_ok=True)
    ROI_mask = load_ROI(ROI_name)
    for subject in SUBJECTS['forrest']:
        movie_data = []
        fns = sorted(glob.glob(f'{indata_dir}/sub-{subject:02d}*movie*')) # loop through all the runs and process separately
        for f in fns:
            data = nib.load(f)
            data_masked = utils.apply_volume_ROI(ROI_mask, data, outfn=False, z=True, zaxis=0)
            movie_data.append(data_masked)
        movie_data = np.concatenate(movie_data, axis=1)
        outfn = f'{outdata_dir}/sub-{subject:02d}_{ROI_name}_movie_all_runs.npy'
        if movie_data.shape[0] != TIMEPOINTS['forrest']:
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
        if loc_data.shape[0] != LOCALIZER_TIMEPOINTS:
            loc_data=loc_data.T
        print(outfn, loc_data.shape)
        np.save(outfn, loc_data)




if __name__ == '__main__':
    # handle both datasets and all ROIs
    for ROI in ROIs:
        mask_sherlock(ROI)
        print(f'finished sherlock {ROI}')
        mask_forrest(ROI)
        print(f'finished forrest {ROI}')














