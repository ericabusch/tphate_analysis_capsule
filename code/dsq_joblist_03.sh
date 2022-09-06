#!/bin/bash
#SBATCH --output log/%A_%2a.out
#SBATCH --array 0-71
#SBATCH --mem-per-cpu 500M -t 24:00:00 -n 1 -c 16 -N 1
#SBATCH --mail-type ALL
#SBATCH --partition=psych_day
#SBATCH --account=casey

# module load miniconda
# source activate /gpfs/milgram/project/casey/elb77/conda_envs/tphate_env 

# DO NOT EDIT LINE BELOW
/gpfs/milgram/apps/hpc.rhel7/software/dSQ/1.05/dSQBatch.py --job-file /gpfs/milgram/project/turk-browne/users/elb77/maneurofold/tphate_analysis_capsule/code/joblist_03.txt --status-dir /gpfs/milgram/pi/turk-browne/users/elb77/maneurofold/tphate_analysis_capsule/code/log
