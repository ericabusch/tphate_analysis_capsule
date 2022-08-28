# utils.py
from nilearn.input_data import NiftiMasker
from nilearn.image import math_img, resample_img
import numpy as np
from scipy.stats import zscore
import pandas as pd
import os

## set up paths
sherlock_dir = '../data/sherlock/'
forrest_dir = '../data/StudyForrest/'
forrest_localizer_labels = '../data/StudyForrest/localizer_labels/'
sherlock_features = '../data/sherlock/behavioral_data/'
forrest_features = '../data/StudyForrest/behavioral_data/'
ROI_dir = '../data/ROIs/'
demo_dir = '../data/demo_data/'

# Variables for each dataset
forrest_subjects = np.array([1,2,3,4,6,9,10,14,15,16,17,18,19,20])
sherlock_subjects = np.arange(1,17)
forrest_movie_timepoints = 3599
sherlock_timepoints = 1976
forrest_localizer_timepoints = 156*4 # four runs of 156 timepoints

ROIs = ['aud_early','early_visual','pmc_nn','high_Visual']
embedding_methods = ['PHATE', 'TPHATE', 'UMAP', 'PCA', 'LLE', 'ISOMAP', 'SMOOTH_PHATE',  "PHATE_TIME", "TSNE"]

################## data loading helper functions #######################
def load_demo_data(ROI): # placeholder argument here to match the others
    """
    Loads in the demo data (which is sherlock early visual)
    :return: list of numpy arrays
    """
    dss=[]
    for s in sherlock_subjects:
        fn = f'{demo_dir}/demo_ROI_data/sub-{s:02d}_early_visual_sherlock_movie.npy'
        d = np.load(fn)
        d = np.nan_to_num(zscore(d, axis=0))
        dss.append(d)
    return dss



def load_sherlock_movie_ROI_data(ROI_name, subjects='all', z=True):
    """
    Loads all data for a given region of interest. This data has already been extracted from the whole-brain
    data into regions of interest.
    :param ROI_name: one of the 4 ROI names
    :param subjects: can be either a list of subject #s from sherlock_subjects or 'all'
    :param z: do you want to zscore the data after loading
    :return: a list of numpy arrays of shape [n_subjects, n_timepoints, n_voxels]
    """
    data_dir = f'{sherlock_dir}/ROI_data/{ROI_name}/data'
    dss = []
    SUBJECTS = sherlock_subjects if subjects == 'all' else subjects
    for s in SUBJECTS:
        fn = f'{data_dir}/sub-{s:02d}_{ROI_name}_sherlock_movie.npy'
        d = np.load(fn)
        d = np.nan_to_num(zscore(d,axis=0)) if z else d
        dss.append(d)
    return dss


def load_forrest_localizer_ROI_data(ROI_name, subjects='all', z=True):
    """
   :param ROI_name: one of the 4 ROI names
   :param subjects: can be either a list of subject #s from sherlock_subjects or 'all'
   :param z: do you want to zscore the data after loading
   :return: a list of numpy arrays of shape [n_subjects, 156*4, n_voxels]
    """
    data_dir = f'{forrest_dir}/ROI_data/{ROI_name}/data'
    dss = []
    SUBJECTS = forrest_subjects if subjects == 'all' else subjects
    for s in SUBJECTS:
        try:
            #these data were already concatenated across run and normalized within run
            fn = f'{data_dir}/sub-{s:02d}_{ROI_name}_localizer_all_runs.npy'
            d = np.load(fn)
        except:
            fn = f'{data_dir}/sub-{s:02d}_{ROI_name}_localizer.npy'
            d = np.load(fn)
        d = np.nan_to_num(zscore(d,axis=0)) if z else d
        dss.append(d)
    return dss


def load_forrest_localizer_labels(sub_id=0, run='all'):
    labels = []
    if run == 'all':
        to_load = [1,2,3,4]
    else:
        to_load = [run]
    if sub_id != 0: # this means we're extracting just a single subject's labels
        fns = [os.path.join(forrest_dir, 'localizer_labels', f'sub-{sub_id:02d}_run_{r}_tr_labels.npy') for r in to_load]
        labels = np.concatenate([np.load(f) for f in fns])
        return labels
    for sub_id in forrest_subjects:
        fns = [os.path.join(forrest_dir, 'localizer_labels', f'sub-{sub_id:02d}_run_{r}_tr_labels.npy') for r in to_load]
        these_labels = np.concatenate([np.load(f) for f in fns])
        labels.append(these_labels)
    return labels




# extract data from a ROI
def apply_volume_ROI(ROI_nii, volume_image, outfn=False, z=True, zaxis=0):
    # if the affine of the mask and volume don't match, resample thhe mask to the volume
    resampled_ROI = resample_img(ROI_nii, target_affine=volume_image.affine, target_shape=volume_image.shape[:-1]).get_fdata()
    resampled_ROI_mask = np.where(resampled_ROI > 0.95, 1, 0) # binarize it
    # now apply the mask to the data
    volume_image=volume_image.get_fdata()
    masked_img = volume_image[resampled_ROI_mask == 1, :]
    if z:
        masked_img = zscore(masked_img, axis=zaxis)
    if outfn != False:
        np.save(outfn, masked_img)
        print(f'saved image of shape {masked_img.shape} to {outfn}')
    return masked_img







################# handle movie annotations for both datasets ############

## this helps for loading binarized, expanded-to-each-timepoint sherlock labels
def load_coded_sherlock_regressors(regressor=None):
    import pandas as pd
    regressors = pd.read_csv(f"{sherlock_features}/sherlock_labels_coded_expanded.csv")
    if regressor:
        regressors = regressors[regressor].values
    return regressors

def load_coded_forrest_movie_regressors(regressor=None):
    import pandas as pd
    regressors = pd.read_csv(f"{forrest_features}/forrest_movie_labels_coded_expanded.csv")
    if regressor:
        regressors = regressors[regressor].values
    return regressors

def label_each_TR(seg_start_TRs, event_onset_TRs, nTRs, nSeg):
    """
    :param seg_start_TRs: list or numpy array of TRs where each new segment of labels begins
    :param event_onset_TRs: the events corresponding with each TR in seg_start_TRs
    :param nTRs: Total number of timepoints that we want to have one label for
    :param nSeg:
    :return: output_labels: a list of labels where one label corresponds for each TR
    """
    if seg_start_TRs[0] != 0:
        nTRs = int(seg_start_TRs[seg_start_TRs.shape[0] - 1] - seg_start_TRs[0])
    output_labels = np.zeros((nTRs,))
    for i in range(0, nSeg - 1):
        curr_TR = seg_start_TRs[i] - seg_start_TRs[0]
        curr_state = event_onset_TRs[i]
        next_TR = seg_start_TRs[i + 1] - seg_start_TRs[0]
        output_labels[int(curr_TR):int(next_TR)] = curr_state
    output_labels[nTRs - 1] = curr_state
    return output_labels


def preprocess_forrest_labels():
    """
    Reads in the original forrest gump annotations files (downloaded from : #TODO insert this here)
    And recodes strings to integer labels for all desired features and the expands to have one label per TR.

    Saves file to the appropriate location for downstream analysis
    and access with `load_coded_forrest_movie_regressors`

    """
    import pandas as pd

    orig_df = pd.read_csv(f"{forrest_features}/ForrestGumpAnnotations.csv")
    orig_df['TR'] = orig_df['time'] // 2  # 2s TR
    # recode interior and exterior : 1 = indoor, 0 = outdoor
    IndoorOutdoor = orig_df['int_or_ext'].values
    IndoorOutdoor[IndoorOutdoor == 'int'] = 1
    IndoorOutdoor[IndoorOutdoor != 1] = 0
    orig_df['IoE_coded'] = IndoorOutdoor

    # recode time of day : 1 = day, 0 = night
    ToD = orig_df['time_of_day'].values
    ToD[ToD == 'day'] = 1
    ToD[ToD != 1] = 0
    orig_df['ToD_coded'] = ToD

    # recode flow of time : 0 = 0, -1 = - , 1 = + , 2 = ++
    FoT = orig_df['flow_of_time'].values
    FoT[FoT == '0'] = 0
    FoT[FoT == '-'] = -1
    FoT[FoT == '+'] = 1
    FoT[FoT == '++'] = 2
    orig_df['FoT_coded'] = FoT

    df_expanded = pd.DataFrame(columns=['time', 'IoE_coded', 'FoT_coded', 'ToD_coded', 'TR'])
    TRs = labels['TR'].astype(int)
    # expand every column
    for col_label in ['time', 'IoE_coded', 'FoT_coded', 'ToD_coded']:
        col_data = labels[col_label].values
        df_expanded[col_label] = label_each_TR(TRs, col_data, 3599, len(labels))
    df_expanded['TR'] = np.arange(len(df_expanded))
    df_expanded.to_csv(f'{forrest_features}/forrest_movie_labels_coded_expanded.csv')




def preprocess_sherlock_labels():
    """
    Reads in the original sherlock movie annotations files (downloaded from : #TODO insert this here)
    And recodes strings to integer labels for all desired features and the expands to have one label per TR.

    Saves file to the appropriate location for downstream analysis
    and access with `load_coded_sherlock_regressors`

    """
    import pandas as pd
    orig_df = pd.read_csv(f"{sherlock_features}/Sherlock_Segments_master.csv")
    orig_df = orig_df[:998] # need to trim this bc lots of blank rows
    nSegments = len(orig_df)
    seg_start_TR = orig_df['Start TR']

    # get rid of nans - expand labels to have one per segment
    without_nans = []
    R_prev = orig_df['Scene Title '].values[0]
    for R in orig_df['Scene Title '].values[:]:
        to_append = ''
        if R != R:
            without_nans.append(R_prev)
        else:
            R_prev = R
            without_nans.append(R)

    # give each scene a code
    unique_scenes = sorted(list(set(without_nans)))
    unique_scenes_coded = [i for i in range(len(unique_scenes))]
    scenes_and_codes = {j: k for j, k in zip(unique_scenes, unique_scenes_coded)}
    without_nans_coded = [scenes_and_codes[scene] for scene in without_nans]

    # valence recoding
    valence_vals = orig_df['valence'].values
    valenceCoded = np.zeros(len(valence_vals))
    valenceCoded[valence_vals == '+'] = 1

    # location recoding
    location_vals = orig_df['Location'].values
    unique_locations = sorted(list(set(location_vals)))
    unique_locations_coded = [i for i in range(len(unique_locations))]
    location_codes = {j: k for j, k in zip(unique_locations, unique_locations_coded)}
    location_coded = [location_codes[loc] for loc in location_vals]

    # indoor/outdoor
    space_vals = orig_df['Space-In/Outdoor'].values
    space_coded = np.zeros(len(space_vals))
    space_coded[space_vals == 'Indoor'] = 1

    # Music present
    music_vals = orig_df['Music Presence '].values
    music_coded = np.zeros(len(music_vals))
    music_coded[music_vals == 'Yes'] = 1

    # now we have one code per segment; expand that to one per TR
    newRegressors['StartTR'] = np.arange(nTR)
    newRegressors['SceneTitleCoded'] = label_each_TR(seg_start_TR, without_nans_coded, nTR, len(seg_start_TR))
    newRegressors['ValenceCoded'] = label_each_TR(seg_start_TR, valenceCoded, nTR, len(seg_start_TR))
    newRegressors['LocationCoded'] = label_each_TR(seg_start_TR, location_coded, nTR, len(seg_start_TR))
    newRegressors['Arousal'] = label_each_TR(seg_start_TR, regressors['Arousal'].values, nTR, len(seg_start_TR))
    newRegressors['MusicPresent'] = label_each_TR(seg_start_TR, music_coded, nTR, len(seg_start_TR))
    newRegressors['IndoorOutdoor'] = label_each_TR(seg_start_TR, space_coded, nTR, len(seg_start_TR))

    CodeValList = list(scenes_and_codes.values())
    SceneKeyList = list(scenes_and_codes.keys())
    newRegressors['SceneTitleString'] = [SceneKeyList[int(code)].strip() for code in
                                         newRegressors['SceneTitleCoded'].values]

    CodeValList = [0, 1]
    ValenceKeyList = ['-', '+']
    newRegressors['Valence'] = [ValenceKeyList[int(code)].strip() for code in newRegressors['ValenceCoded'].values]

    CodeValList = list(location_codes.values())
    LocationKeyList = list(location_codes.keys())
    newRegressors['Location'] = [LocationKeyList[int(code)].strip() for code in newRegressors['LocationCoded'].values]
    newRegressors['SceneTitleCoded'] = newRegressors['SceneTitleCoded'].astype(int)
    newRegressors['ValenceCoded'] = newRegressors['ValenceCoded'].astype(int)
    newRegressors['LocationCoded'] = newRegressors['LocationCoded'].astype(int)
    newRegressors['Arousal'] = newRegressors['Arousal'].astype(int)
    newRegressors.to_csv(f'{sherlock_features}/sherlock_labels_coded_expanded.csv')
