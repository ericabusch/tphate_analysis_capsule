"""
step_09_demap_simulations.py

Script to create the data that is then used for simulations and demap
analysis.
"""

import numpy as np
import pandas as pd
import demap, phate, scprep
import embedding_helpers as mf
import utils
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

np.random.seed(64)

def f(x, alpha, sigma):
    """
    Function for generating signal with a known autocorrelation function and varying
    amounts of noise (noise sampled from a normal distribution of SD=sigma, according to:
    f(X_t) = alpha * f(X_t-1) + ε
    ε ∼ Ν(0, σ2)

    :returns
    pristine_data: ground truth matrix (pristine autocorrelated data)
    simulated_data: pristine_data with added normally distributed noise

    """
    pristine_data = alpha * x
    eps = np.random.normal(0.0, sigma, size=x.shape)
    simulated_data=pristine_data + eps
    return pristine_data, simulated_data

embed_functions = {"LLE": demap.embed.LLE,
                   "ISOMAP":demap.embed.Isomap,
                   "TSNE":demap.embed.TSNE,
                   "PHATE":demap.embed.PHATE,
                   "PCA":demap.embed.PCA,
                   "UMAP":demap.embed.UMAP,
                   "TPHATE":mf.embed_tphate
                   }
samples = 40
features = 20
alphas = [4, 5, 6]
sigmas = [0.5, 1, 2, 3, 5, 10, 20, 50 , 80, 100]

demap_df = pd.DataFrame(columns=['method', 'sigma', 'alpha', 'rho'])

X = np.ones((samples, features)) * np.arange(samples).reshape(-1, 1)
for alpha in alphas:
    for sigma in sigmas:
        Ys = np.array([f(x, alpha, sigma) for x in X])
        Y_true, Y_noise = Ys[:,0,:], Ys[:,1,:]
        for meth, func in embed_functions.items():
            Y_embd = func(Y_noise)
            score = demap.DEMaP(Y_true, Y_embd)
            demap_df.loc[len(demap_df)] = {'method': meth, "sigma": sigma, 'alpha': alpha, 'rho': score}
    print(sigma, alpha)

demap_df.to_csv('../results/demap_simulation_results.csv')

