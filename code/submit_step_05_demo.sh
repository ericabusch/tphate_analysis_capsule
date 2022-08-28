#!/bin/bash
#SBATCH --output log/%A.o
#SBATCH --job-name demo_step5
#SBATCH --mem-per-cpu 5G -t 10:00:00 -n 1 -c 1
#SBATCH --mail-type ALL
#SBATCH --partition=psych_day
#SBATCH --account=turk-browne

source activate /gpfs/milgram/project/casey/elb77/conda_envs/tphate_env

python -u step_05_WvB_behavior_boundaries.py early_visual demo