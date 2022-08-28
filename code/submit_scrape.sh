#!/bin/bash
#SBATCH --output log/%A.o
#SBATCH --job-name scrape_files
#SBATCH --mem-per-cpu 5G -t 00:10:00 -n 1 -c 1
#SBATCH --mail-type ALL
#SBATCH --partition=psych_day
#SBATCH --account=turk-browne

source activate /gpfs/milgram/project/casey/elb77/conda_envs/tphate_env

SCRIPT=$1
first="4p5"

echo $SCRIPT
if [ "$SCRIPT" = "$first" ] ; then
  echo "Running step_04p5"
  python -u step_04p5_scrape_HMM_results.py demo
else
  echo "Running step_06p5"
  python -u step_06p5_scrape_HMM_results.py demo
fi
