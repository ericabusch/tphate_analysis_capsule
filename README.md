# T-PHATE Analysis Capsule

**T-PHATE manifold learning of brain state dynamics**

This is a code capsule comprising all scripts needed to replicate the analyses in "Multi-view manifold learning of human brain state trajectories" (Busch et al., 2022, [bioRxiv](https://www.biorxiv.org/content/10.1101/2022.05.03.490534v3)). In this paper, we present T-PHATE (_temporal potential of heat diffusion for affinity-based transition embedding_), a novel multi-view manifold learning algorithm. We demonstrate how T-PHATE learns meaningful low-dimensional representations of neural data that can be used to track static and dynamic properties of brain states during multiple fMRI tasks. We benchmark T-PHATE against 6 of the most commonly used dimensionality reduction methods in biomedical sciences, as well as 2 alternative implementations of T-PHATE, and the full, high-dimensional representation. Through denoising brain data in both space and time, T-PHATE yields a latent structure in fMRI data that better represents static and dynamic information and corresponds with human behavior, allowing enhanced access at the neural signatures of higher-order cognition.  

Since the data needed to replicate all analyses entirely are too large to upload and share on this platform, and would take too long to run without HPC access and substantial parallelization, we provide demo fMRI data for a single region of interest (in `data/demo_data/demo_ROI_data`) and pre-computed T-PHATE embeddings (in `data/demo_data/demo_embeddings`) replicate analyses for a single dataset, region of interest, and embedding algorithm (T-PHATE). One can run all analyses for the demo data by including `demo` as the final command-line argument for each of the scripts. Scripts can be run for the demo as outlined below. A conda environment containing all requirements is available in `environment.yml`. 

The T-PHATE algorithm is available as a python package [here](https://github.com/KrishnaswamyLab/TPHATE).

## Running demo pipeline
Demo data are provided for one of four ROIs in one of two datasets (early visual ROI, _sherlock_ dataset). We provide the TPHATE embeddings needed for each of these files to complete the following analyses, which could be generated from the original data using `step_00_apply_ROIs.py` and `step_01_run_embeddings.py` for all other methods/regions/datasets.

Parameters set for the demo pipeline:
DATASET=demo
ROI=early_visual
METHOD=TPHATE

To run the demo pipeline:
1. `python step_02_HMM_optimizeK_voxel.py $DATASET $ROI`
    - Fits and tests HMMs for event segmentation with leave-one-subject-out cross validation on voxel-resolution data to select HMM hyperparameter __K__ for each subject, to be used in fitting HMMs on dimensionality-reduced data in subsequent analyses.
    - Saves files `intermediate_data/demo/HMM_learnK/{DATASET}_{ROI}_samplingK_LOSO.csv` and `intermediate_data/demo/HMM_learnK/{DATASET}_{ROI}_bestK_LOSO.csv`.
2. `python step_03_HMM_optimizeM_embeddings.py $DATASET $ROI ` 
    - Takes the `${DATASET}_${ROI}_bestK_LOSO.csv` file and scrapes it to get the best K value (# neural events identified by HMM for a given region). Uses that value to re-fit HMMs to reduced-dimension embeddings of the neural data for the region and evaluate within-vs-between event-boundary distances, then choose the number of embedding dimensions __M__ that maximizes the within-vs-between event-boundary distances. 
    - The __M__ value learned for each subject is then cross-validated across subjects in the same fashion as the __K__ hyperparameter. These are then used to fit a final HMM per subject and compute the within-vs-between event-boundary distance & model fit reported in figures 4 and 5. 
3. `python step_04_HMM_WvB_boundaries_voxel.py $DATASET $ROI`
    - Performs the same analysis as in `step_03_HMM_optimizeM_embeddings` but without the optimization of the number of dimensions __M__, instead keeping the full voxel-resolution dataset.
4. `python step_04p5_scrape_HMM_results.py` 
    - When running the full version, scrapes all the results of step 2 and 3 across ROIs and datasets into one file for convenience in future analyses. 
5. `python step_05_WvB_behavior_boundaries.py $DATASET $ROI` 
    - Instead of using HMMs to identify event boundaries as in steps 2 and 3, this analysis uses event boundaries identified by a separate cohort of human raters. Applies those boundaries to the voxel resolution and embedding data and then evaluates their fit as measured in previous analyses. 

