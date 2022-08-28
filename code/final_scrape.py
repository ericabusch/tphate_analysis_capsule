"""
final_scrape.py

This script just scrapes together the results from the previous two steps into one csv file for
easy access in subsequent steps.

"""


import numpy as np
import pandas as pd
import os, sys, glob


if len(sys.argv) > 1:
    demo = True
    indir = '../results/demo_results/source/'
    outdir='../results/demo_results/'
else:
    demo=False
    indir='../results/source'
    outdir = '../results'

# scrape behavior results
fns = glob.glob(indir+'/*behavior_event_boundaries_WB_tempBalance.csv')
outfn_name = f'{outdir}/behavioral_event_WB_tempBalance_results.csv'
results = pd.concat([pd.read_csv(f,index_col=0) for f  in fns])
results.to_csv(outfn_name)
print(outfn_name)

# scrape between subj results
fns = glob.glob(indir+'/*between_sub_neural_event_WB_tempBalance.csv')
outfn_name = f'{outdir}/between_sub_neural_event_WB_tempBalance_results.csv'
results = pd.concat([pd.read_csv(f,index_col=0) for f  in fns])
results.to_csv(outfn_name)
print(outfn_name)

# scrape SVC results
fns = glob.glob(indir+'/*SVC_movie_zstat_results.csv')
outfn_name = f'{outdir}/SVC_movie_zstat_results.csv'
results = pd.concat([pd.read_csv(f,index_col=0) for f  in fns])
results.to_csv(outfn_name)
print(outfn_name)

if not demo:
    # scrape localizer SVC results
    fns = glob.glob(indir+'/*SVC_localizer_results.csv')
    outfn_name = f'{outdir}/SVC_localizer_results.csv'
    results = pd.concat([pd.read_csv(f,index_col=0) for f  in fns])
    results.to_csv(outfn_name)
    print(outfn_name)