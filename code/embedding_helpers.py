import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding, TSNE
import phate, umap
from TPHATE.tphate import tphate
import os
import statsmodels.api as sm


def return_subject_embedding(embed_fn, timeseries_data, manifold_dim, manifold_method, return_embd=True):
    """
    :param embed_fn: (string) The filename of the embedding you're looking for
    :param timeseries_data: (np.array) The [samples x features] matrix to embed
    :param manifold_dim: (int) the number of dimensions for embedding
    :param manifold_method: (string) the method to use for embedding (one of the keys in the func_dict)
    :return: The embedding

    Searches for a filename if it exists, otherwise creates the embedding and saves it. Returns the embedding as well.
    """
    if os.path.exists(embed_fn):
        return np.load(embed_fn)

    func = func_dict[manifold_method]
    embed = func(timeseries_data, manifold_dim)
    print(f'Saving {embed_fn} | shape: {embed.shape}')
    np.save(embed_fn, embed)
    if return_embd:
        return embed
    
# smooth PHATE with kernel width of TPHATE
def embed_smoothed_phate(data, n_components):
    new_data = np.zeros_like(data)
    acf = sm.tsa.acf(np.nanmean(data,axis=1), fft=False, nlags=data.shape[0]-1, missing='drop')
    dropoff=np.where(acf<0)[0][0]
    for i in range(data.shape[1]):
        new_data[:, i] = np.convolve(data[:, i], np.ones(dropoff), 'same') / dropoff
    embedding = phate.PHATE(n_components=n_components, n_jobs=-1, verbose=0, n_landmark=data.shape[0]).fit_transform(
        new_data)
    return embedding

# Smooth PHATE implementation (set to 2 to match HRF)
def embed_smoothed_HRF_phate(data, n_components, smooth_window=2):
    new_data = np.zeros_like(data)
    for i in range(data.shape[1]):
        new_data[:, i] = np.convolve(data[:, i], np.ones(smooth_window), 'same') / smooth_window
    embedding = phate.PHATE(n_components=n_components, n_jobs=-1, verbose=0, n_landmark=data.shape[0]).fit_transform(
        new_data)
    return embedding

# PHATE + Time
def embed_phate_time(data, n_components):
    new_data = np.hstack((data, np.arange(data.shape[0]).reshape(-1, 1)))
    embd = phate.PHATE(verbose=0, n_jobs=-1, n_landmark=new_data.shape[0], n_components=n_components).fit_transform(
        new_data)
    return embd

# TPHATE
def embed_tphate(data, n_components=2):
    embed = tphate.TPHATE(verbose=0, n_jobs=-1, n_landmark=data.shape[0],
                        n_components=n_components, t=5).fit_transform(data)
    return embed

# PHATE
def embed_phate(data, n_components):
    embedding = phate.PHATE(n_components=n_components, n_jobs=-1, verbose=0, n_landmark=data.shape[0]).fit_transform(
        data)
    return embedding

# PCA
def embed_pca(data, n_components):
    embedding = PCA(n_components=n_components).fit_transform(data)
    return embedding

# Locally Linear Embedding
def embed_lle(data, n_components):
    embedding = LocallyLinearEmbedding(n_components=n_components).fit_transform(data)
    return embedding

# UMAP
def embed_umap(data, n_components):
    embedding = umap.UMAP(n_neighbors=5, n_components=n_components).fit_transform(data)
    return embedding

# Isomap
def embed_isomap(data, n_components):
    embedding = Isomap(n_components=n_components).fit_transform(data)
    return embedding

# TSNE
def embed_tsne(data, n_components):
    embedding = TSNE(n_components=n_components).fit_transform(data)
    return embedding


func_dict = {'PHATE': embed_phate, 'TPHATE': embed_tphate, 'UMAP': embed_umap, 'PCA': embed_pca,
             'LLE': embed_lle, 'ISOMAP': embed_isomap, 'SMOOTH_PHATE': embed_smoothed_phate,
             "PHATE_TIME": embed_phate_time, "TSNE": embed_tsne}
