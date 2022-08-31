import numpy as np
import utils, config
import embedding_helpers as mf
from joblib import Parallel, delayed
from config import NJOBS


def drive_embeddings():
    original_data = np.squeeze(LOADFN(ROI, subjects=[SUBJECT], z=True))

    joblist = []
    for method in config.EMBEDDING_METHODS:
        dim2embed=config.DIMENSIONS_TO_TEST if method != 'TSNE' else [2,3] # tsne can only go to 3
        for n_dim in dim2embed:
            filename = f'{EMBED_DIR}/sub-{SUBJECT:02d}_{ROI}_{DATASET}_movie_{n_dim}dimension_embedding_{method}.npy'
            joblist.append((delayed(mf.return_subject_embedding)(filename, original_data, n_dim, method, return_embd=False)))

    with Parallel(n_jobs=NJOBS) as parallel:
        parallel(joblist)

def drive_localizer_embeddings():
    original_data= np.squeeze(utils.load_forrest_localizer_ROI_data(ROI, subjects=[SUBJECT], z=True))
    joblist = []
    for method in config.EMBEDDING_METHODS:
        for n_dim in [2,10,20]:
            filename = f'{EMBED_DIR}/sub-{SUBJECT:02d}_{ROI}_{DATASET}_localizer_{n_dim}dimension_embedding_{method}.npy'
            joblist.append(
                (delayed(mf.return_subject_embedding)(filename, original_data, n_dim, method, return_embd=False)))

    with Parallel(n_jobs=NJOBS) as parallel:
        parallel(joblist)


if __name__ == "__main__":
    DATASET = sys.argv[1]
    ROI = sys.argv[2]
    SUBJECT = sys.argv[3]

    NJOBS = 16
    LOADFN = utils.LOAD_FMRI_FUNCTIONS[DATASET]
    drive_embeddings()

    if DATASET=='forrest':
        drive_localizer_embeddings()

