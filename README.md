# T-PHATE Analysis Capsule

**T-PHATE manifold learning of brain state dynamics**

This is a code capsule comprising all scripts needed to replicate the analyses in "Multi-view manifold learning of human brain state trajectories" (Busch et al., 2022, [bioRxiv](https://www.biorxiv.org/content/10.1101/2022.05.03.490534v3)). In this paper, we present T-PHATE (_temporal potential of heat diffusion for affinity-based transition embedding_), a novel multi-view manifold learning algorithm. We demonstrate how T-PHATE learns meaningful low-dimensional representations of neural data that can be used to track static and dynamic properties of brain states during multiple fMRI tasks. We benchmark T-PHATE against 6 of the most commonly used dimensionality reduction methods in biomedical sciences, as well as 2 alternative implementations of T-PHATE, and the full, high-dimensional representation. Through denoising brain data in both space and time, T-PHATE yields a latent structure in fMRI data that better represents static and dynamic information and corresponds with human behavior, allowing enhanced access at the neural signatures of higher-order cognition.  

Since the data needed to replicate all analyses entirely are too large to upload and share on this platform, and would take too long to run without HPC access and substantial parallelization, we provide demo fMRI data for a single region of interest (in `data/demo_data/demo_ROI_data`) and pre-computed T-PHATE embeddings (in `data/demo_data/demo_embeddings`) replicate analyses for a single dataset, region of interest, and embedding algorithm (T-PHATE). One can run all analyses for the demo data by including `demo` as the final command-line argument for each of the scripts. Scripts can be run for the demo as outlined below. A conda environment containing all requirements is available in `environment.yml`. 

The T-PHATE algorithm is available as a python package [here](https://github.com/KrishnaswamyLab/TPHATE).

## Running demo pipeline
Demo data are provided for one of four ROIs in one of two datasets (early visual ROI, _sherlock_ dataset). We provide the TPHATE embeddings needed for each of these files to complete the following analyses, which could be generated from the original data using `step_00_apply_ROIs.py` and `step_01_run_embeddings.py` for all other methods/regions/datasets.

Set following variable for demo:   
DATASET=sherlock      
ROI = early_visual   
METHOD=TPHATE   
DEMO=demo


1. `python step_02_HMM_optimizeK_voxel.py sherlock early_visual demo`
    - Fits and tests HMMs for event segmentation with leave-one-subject-out cross validation on voxel-resolution data to select HMM hyperparameter `K` for each subject, to be used in fitting HMMs on dimensionality-reduced data in subsequent analyses.
    - Saves files `intermediate_data/demo/HMM_learnK/{DATASET}_{ROI}_samplingK_LOSO.csv` and `intermediate_data/demo/HMM_learnK/{DATASET}_{ROI}_bestK_LOSO.csv`.
2. `python step_03_HMM_optimizeM_embeddings.py $DATASET $ROI $METHOD $DEMO` 
    - Takes the `${DATASET}_${ROI}_bestK_LOSO.csv` file and scrapes it to get the best K value (# neural events identified by HMM for a given region) and uses that value to re-fit HMMs to the TPHATE embeddings for that region.
