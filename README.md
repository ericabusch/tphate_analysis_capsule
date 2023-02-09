# T-PHATE Analysis Capsule

[![DOI](https://zenodo.org/badge/529971243.svg)](https://zenodo.org/badge/latestdoi/529971243)


**T-PHATE manifold learning of brain state dynamics**

This is a code capsule comprising all scripts needed to replicate the analyses in "Multi-view manifold learning of human brain state trajectories" (Busch et al., 2022, [bioRxiv](https://www.biorxiv.org/content/10.1101/2022.05.03.490534v4)). In this paper, we present T-PHATE (_temporal potential of heat diffusion for affinity-based transition embedding_), a novel multi-view manifold learning algorithm. We demonstrate how T-PHATE learns meaningful low-dimensional representations of neural data that can be used to track static and dynamic properties of brain states during multiple fMRI tasks. We benchmark T-PHATE against 6 of the most commonly used dimensionality reduction methods in biomedical sciences, as well as 2 alternative implementations of T-PHATE, and the full, high-dimensional representation. Through denoising brain data in both space and time, T-PHATE yields a latent structure in fMRI data that better represents static and dynamic information and corresponds with human behavior, allowing enhanced access at the neural signatures of higher-order cognition.  

Since the data needed to replicate all analyses entirely are too large to upload and share on this platform, and would take too long to run without HPC access and substantial parallelization, we provide demo fMRI data for a single region of interest (in `data/demo_data/demo_ROI_data`) and pre-computed T-PHATE embeddings (in `data/demo_data/demo_embeddings`) replicate analyses for a single dataset, region of interest, and embedding algorithm (T-PHATE). Scripts can be run for the demo as outlined below. We also include the results for the demo analysis in `results/demo_results`. A conda environment containing all requirements is available in `environment.yml`. 

The T-PHATE algorithm is available as a python package [here](https://github.com/KrishnaswamyLab/TPHATE).

## Running demo pipeline
Demo data are provided for one of four ROIs in one of two datasets (early visual ROI, _sherlock_ dataset). We provide the TPHATE embeddings needed for each of these files to complete the following analyses, which could be generated from the original data using `step_00_apply_ROIs.py` and `step_01_run_embeddings.py` for all other methods/regions/datasets.

Parameters set for the demo pipeline:      
DATASET=demo      
ROI=early_visual     
METHOD=TPHATE       

To run the demo pipeline:
- `python step_02_HMM_optimizeK_voxel_nestedCV_parallel_by_sub.py $DATASET $ROI`
    - Fits and tests HMMs for event segmentation with nested leave-one-subject-out cross validation on voxel-resolution data to select HMM hyperparameter _K_ for each subject, to be used in fitting HMMs on dimensionality-reduced data in subsequent analyses.
    - Saves files `intermediate_data/demo/HMM_learnK/{DATASET}_{ROI}_samplingK_nestedCV_validation_sub-{VALIDATION_SUBJECT}.csv` and `intermediate_data/demo/HMM_learnK/{DATASET}_{ROI}_bestK_nestedCV_validation_sub-{VALIDATION_SUBJECT}.csv`.
    - Will default to using `VALIDATION_SUBJECT=0`
- `python step_03_HMM_embeddings.py $DATASET $ROI ` 
    - Takes the `${DATASET}_${ROI}_bestK_nestedCV_validation_sub-{validation_subject}.csv` file and scrapes it to get the best K value (# neural events identified by HMM for a given region). Uses that value to re-fit HMMs to reduced-dimension embeddings of the neural data for the region and evaluate within-vs-between event-boundary distances, then choose the number of embedding dimensions _M_ that maximizes the within-vs-between event-boundary distances. 
    - The _M_ value learned for each subject is then cross-validated across subjects. These are then used to fit a final HMM per subject and compute the within-vs-between event-boundary distance & model fit (Results reported in Figure 5, Supplementary Figures 4, 5A, 6A). 
    - Also runs a control analysis repeating the same analyses with a fixed _M_=3 embedding dimensionality (presented in Supplementary Figures 7,8).
- `python step_04_HMM_WvB_boundaries_voxel.py $DATASET $ROI`
    - Performs the same analysis as in `step_03_HMM_optimizeM_embeddings` but without the optimization of the number of dimensions _M_, instead keeping the full voxel-resolution dataset (Results reported in Figure 5).
- `python step_04p5_scrape_HMM_results.py` 
    - When running the full version, scrapes all the results of step 2 and 3 across ROIs and datasets into one file for convenience in future analyses. 
- `python step_05_WvB_behavior_boundaries.py $DATASET $ROI` 
    - Instead of using HMMs to identify event boundaries as in steps 2 and 3, this analysis uses event boundaries identified by a separate cohort of human raters. Applies those boundaries to the voxel resolution and embedding data and then evaluates their fit as measured in previous analyses (Figure 6B). Also has a control analysis to repeat this procedure with a constant _M_=3 (Supplementary Figure 8).
    - This can only be done for the _sherlock_ dataset as the _forrest_ dataset does not have equivalent ratings.
- `python step_06_WvB_boundaries_cross_subjects.py $DATASET $ROI $METHOD`
    - Using the event boundaries identified with HMMs within-subject that were saved from step 3 and step 4, this script evaluates the fit of these boundaries _across_ subjects. Since we'd expect a degree of coherence between-subjects, we can ask how well the boundaries learned within subject extend to another subject based on embedding method.
- `python step_07_SVC_movie_features.py $DATASET $ROI $METHOD`
    - Using labels for each TR of the movie stimuli, classifies features of the movie using a support vector classifier, and normalizes them to time-shifted versions of their labels (reported in figure 2B and S3).
- `python step_08_SVC_localizer.py $METHOD`
    - Takes the localizer stimulus labels and runs a SVC for two visual ROIs for the forrest localizer dataset (figures 3C and S1B).
- `python step_09_demap_simulations.py`
    - Generates simulated data and replicates results for the analysis in figure 1 and figure S1A, to show denoised manifold preservation.
 
Additional scripts are included as helpers for housing common functions, variable names, and SLURM submission scripts. Command line arguments change with scripts to speed up run-time and assist with memory allocation in job submission.

## Requirements
This software was implemented and run on a Red Hat Linux Compute Cluster ("Red Hat Enterprise Linux Server 7.9 (Maipo)"). The demo was additionally tested on a macOS Monterey version 12.4. The conda environment specified in `environment.yml` contains all dependencies to recreate our compute environment. The environment can be created using `conda env create environment.yml` with anaconda (tested with conda version 4.10.1 on macOS and conda 4.12.0 on Linux); took around 16 minutes to create environment `tphate_environment_macos.yml` on local macOS. The scripts here were run with parallelization both within-script and at the job level on our HPC cluster using dead-simple queue. 
