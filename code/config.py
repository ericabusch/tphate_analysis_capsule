from os.path import *
import numpy as np

NJOBS=16

ROOT = '/gpfs/milgram/scratch60/turk-browne/neuromanifold'

RAW_DATA_FOLDERS = {'sherlock':join(ROOT, "sherlock/MNI152_3mm_data/denoised_filtered_smoothed/whole_brain_data/")
                    'forrest':join(ROOT, "StudyForrest/MNI152_3mm_data/denoised_filtered_smoothed/whole_brain_data/"}

DATA_FOLDERS={'demo':'../data/demo_data/',
            'sherlock':'../data/sherlock/',
           'forrest':'../data/StudyForrest/'}

FEATURES_FILES = {'demo':'../data/sherlock/behavioral_data/sherlock_labels_coded_expanded.csv',
                   'sherlock':'../data/sherlock/behavioral_data/sherlock_labels_coded_expanded.csv',
                   'forrest':'../data/StudyForrest/behavioral_data/forrest_movie_labels_coded_expanded.csv'}

FEATURES_FILES_ORIGINAL = {'sherlock':'../data/sherlock/behavioral_data/Sherlock_Segments_master.csv',
                           'forrest':'../data/StudyForrest/behavioral_data/ForrestGumpAnnotations.csv'}

INTERMEDIATE_DATA_FOLDERS={'demo':'../intermediate_data/demo',
                           'sherlock':'../intermediate_data',
                           'forrest':'../intermediate_data'}

RESULTS_FOLDERS={'demo':'../results/demo_results',
                           'sherlock':'../results',
                           'forrest':'../results'}




LOCALIZER_FOLDER = '../data/StudyForrest/localizer_labels'

ROIs = ['aud_early','early_visual','pmc_nn','high_Visual']

EMBEDDING_METHODS = ['PHATE', 'TPHATE', 'UMAP', 'PCA', 'LLE',
                     'ISOMAP', 'SMOOTH_PHATE',  "PHATE_TIME", "TSNE"]

ROI_FILES = {'aud_early':'../data/ROIs/aud_early.nii',
             'early_visual':'../data/ROIs/early_visual.nii',
             'high_Visual':'../data/ROIs/high_Visual.nii.gz',
             'pmc_nn.nii':'../data/ROIs/pmc_nn.nii'}

SUBJECTS  = {'demo': list(np.arange(1,17)),
             'sherlock': list(np.arange(1,17)),
             'forrest': [1,2,3,4,6,9,10,14,15,16,17,18,19,20]}

TIMEPOINTS = {'demo':1976,
              'sherlock':1976,
              'forrest':3599}

LOCALIZER_TIMEPOINTS=156*4

HMM_K_TO_TEST = {'demo':np.arange(10, 121, 1),
                 'sherlock': np.arange(10, 121, 1),
                 'forrest':np.arange(20, 201, 1)}
DIMENSIONS_TO_TEST = np.arange(2, 11)