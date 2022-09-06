# scrape together all the files outputted from step 02
# into one large dataframe

import os, sys, glob
import utils, config
import pandas as pd

for dataset in config.DATASETS:
    input_dir = config.INTERMEDIATE_DATA_FOLDERS[dataset]+'/HMM_learnK_nested'
    for roi in config.ROIs:
        df = pd.DataFrame(columns=["ROI","dataset","test_subject","CV_K"])
        for subject in config.SUBJECTS[dataset]:
            fn = f'{input_dir}/{dataset}_{roi}_bestK_nestedCV_validation_sub-{subject:02d}.csv'
            temp = pd.read_csv(fn, index_col=0)
            # renaming to match formatting in subsequent scripts
            temp.rename({'bestK':"CV_K", "validation_subject":"test_subject"}, axis=1, inplace=True)
            df = pd.concat([df,temp])
        df.to_csv(f'{input_dir}/{dataset}_{roi}_bestK_nestedCV.csv')
        print(f'done {dataset} {roi} {df.CV_K.mean(), df.CV_K.std()}')