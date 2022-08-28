"""
step_09_demap_simulations.py

Script to create the data that is then used for simulations and demap
analysis, then create plots comparable to figure 1B and figure s1A
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
alphas = [3, 4, 5]
sigmas = [0.5, 1, 2, 3, 50 , 80, 100]#[0.5, 1, 2, 3, 5, 10, 20, 50 , 80, 100]

demap_df = pd.DataFrame(columns=['method', 'sigma', 'alpha', 'rho'])

X = np.ones((samples, features)) * np.arange(samples).reshape(-1, 1)
embeddings = {}
for alpha in alphas:
    for sigma in sigmas:
        Ys = np.array([f(x, alpha, sigma) for x in X])
        Y_true, Y_noise = Ys[:,0,:], Ys[:,1,:]
        for meth, func in embed_functions.items():
            Y_embd = func(Y_noise)
            embeddings[f'{meth}_{alpha}_{sigma}'] = Y_embd
            score = demap.DEMaP(Y_true, Y_embd)
            demap_df.loc[len(demap_df)] = {'method': meth, "sigma": sigma, 'alpha': alpha, 'rho': score}

## plot all embedding methods for each end of the noise extreme
order=['PCA','LLE','ISOMAP','TSNE','UMAP','PHATE','TPHATE']
top_keys = [f'{meth}_4_0.5' for meth in order]
bottom_keys = [f'{meth}_4_100' for meth in order]
fig,ax=plt.subplots(2,8,figsize=(16, 4))
X = np.ones((10,10))*np.arange(10)

# lower sigma matrix
Ys = np.array([f(x, 4, 0.5) for x in X]).T
ax[0, 0].imshow(Ys[:,1,:])
ax[0,0].set_xticks([])
ax[0,0].set_yticks([])
for i, key in enumerate(top_keys):
    title=key.split("_")[0]
    scprep.plot.scatter2d(embeddings[key],
                          c=np.arange(samples),
                          ticks=False,
                          title=title,
                          edgecolor='gray',
                          ax=ax[0,i+1],
                          legend=False,
                          cmap='YlGnBu_r')

# higher sigma matrix
Ys = np.array([f(x, 5, 100) for x in X])
ax[1,0].imshow(Ys[:,1,:])
ax[1,0].set_xticks([])
ax[1,0].set_yticks([])
for i, key in enumerate(bottom_keys):
    title=key.split("_")[0]
    scprep.plot.scatter2d(embeddings[key],
                          c=np.arange(samples),
                          ticks=False,
                          title=title,
                          edgecolor='gray',
                          ax=ax[1,i+1],
                          legend=False,
                          cmap='YlGnBu_r')
plt.savefig('../plots/demap_simulation_scatterplots.png',
            bbox_inches = "tight",
            transparent=True) # save as png for space
plt.close()

# now plotting the line plot
fig,ax=plt.subplots(figsize=(6,6))
g=sns.lineplot(data=demap_df, x='sigma', y='rho', hue='method', ci=10, # just for here, setting this really narrow
               hue_order=order[::-1], palette='magma')
plt.legend(labels=order[::-1], title = "",
           fontsize = 'x-large', loc='lower left')
g.set_ylabel("DeMAP",Fontsize="large")
g.set_xlabel("Sigma",Fontsize="large")
g.set(ylim=[0,1], xlim=[0,100])
plt.savefig('../plots/demap_simulation_lineplot.png',
            bbox_inches = "tight",
            transparent=True)
plt.close()
demap_df.to_csv('../results/demap_simulation_results.csv')
