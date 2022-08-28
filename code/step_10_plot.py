"""
step_10_plot.py
Plot all results

"""
import numpy as np
import pandas as pd
import os,sys,glob
import matplotlib.pyplot as plt
import seaborn as sns
import utils
import scprep, phate
import matplotlib as mpl
import matplotlib.patches as patches
import scipy.stats as stats
from TPHATE.tphate import tphate

# set color palettes
evis_colors = ["#FFFFFF","#E9B0D2", "#DE88BC", "#D360A5", "#AF4F88"]
hvis_colors = ["#FFFFFF","#CCB2D5", "#B28BC0","#9864AB","#7D538D"]
aud_colors = ["#FFFFFF",'#D1F0F2','#A2E1E4',"#44C2C9","#38A0A6"]
pmc_colors = ["#FFFFFF","#B1DFB1","#8DD18D","#58BB58","#499B49"]
big_colors = ["#AF4F88", "#7D538D", "#38A0A6", "#499B49"]

evis_3colors = ["#E9B0D2", "#DE88BC", "#AF4F88"]
hvis_3colors = ["#CCB2D5", "#B28BC0","#7D538D"]
aud_3colors = ['#D1F0F2','#A2E1E4',"#38A0A6"]
pmc_3colors = ["#B1DFB1","#8DD18D","#499B49"]

# keywords for making pdfs
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Figure 1A
import statsmodels.api as sm
data=utils.load_demo_data()[6] # get subject 7's demo data

# Autocorrelation function
A_vox = np.empty_like(data)
n_timepoints, n_voxels = data.shape
for v in range(n_voxels):
    x = data[:, v]
    A_vox[:, v] = sm.tsa.acf(x, fft=False, nlags=len(x)-1)
A_avg = np.mean(A_vox, axis=1)
y = np.convolve(A_avg, np.ones(smooth_window), 'same') / smooth_window
dropoff=np.where(y < 0)[0][0]
sns.set(style='white')
fig,ax=plt.subplots(1,1,figsize=(5,4))
_=ax.plot(A_vox[1:56], alpha=0.05, c='lightpink')
ax.plot(y[1:56], c='maroon')
ax.axhline(0, color='k', linestyle='--')
sns.despine(top=True,right=True)
ax.set(xlim=[0,20], xlabel='Lags', ylabel="Autocorrelation")
ax.set(xticks=np.arange(0,21,10), yticks=[-0.25,0,0.25,0.5,0.75,1.0], yticklabels=[-0.25,0,'',0.5,'',1.0])
ax.plot([dropoff,dropoff],[-0.25,1.0],'gray',ls='--',linewidth=3)
plt.savefig('../plots/figure1A_line.png')
plt.close()

# fmri activity mtx
data=data[100:180, 200:300]
fig,ax=plt.subplots(figsize=(8,8))
g=sns.heatmap(data,square=True,xticklabels=False,yticklabels=False,cbar=False )
g.set_ylabel("Timepoints", fontsize=30)
g.set_xlabel("Voxels", fontsize=30)
g.set_title("fMRI activity",fontsize=40)
plt.savefig('../plots/figure1A_fmri_activity.png')

# fit tphate

# autocorrelation view


