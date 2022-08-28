"""
step_04p5_scrape_HMM_results.csv

This script just scrapes together the results from the previous two steps into one csv file for
easy access in subsequent steps.

"""


import numpy as np
import pandas as pd
import os, sys, glob


if sys.argv[1] == 'demo':
    indir = '../results/demo_results/source/'
    outdir='../results/demo_results/'
else:
    indir='../results/source'
    outdir = '../results'

fns = glob.glob(indir+'/*tempBalance_crossValid_WB_results.csv')
outfn_name = f'{outdir}/within_sub_neural_event_WB_tempBalance_results.csv'
results = pd.concat([pd.read_csv(f,index_col=0) for f  in fns])
results.to_csv(outfn_name)
