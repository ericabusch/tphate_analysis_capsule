#!/bin/bash
#SBATCH --output log/%A.o
#SBATCH --job-name demo_step2
#SBATCH --mem-per-cpu 5G -t 24:00:00 -n 1 -c 16
#SBATCH --mail-type ALL
#SBATCH --partition=psych_day
#SBATCH --account=turk-browne

source activate /gpfs/milgram/project/casey/elb77/conda_envs/tphate_env 

python -u step_02_HMM_optimizeK_voxel.py sherlock early_visual demo
