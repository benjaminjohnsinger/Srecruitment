#!/bin/bash
#SBATCH --job-name=rcRSVrrp
#SBATCH --account=ac_idmodels
#SBATCH --partition=savio2
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --array=0-3
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bjsinger@berkeley.edu
## Command(s) to run:

module load python/3.11.6-gcc-11.4.0

combinations=(
"RSV 250311 Stepwise X X 15 1 0.7"
"RSV 250311 Stepwise nb X 15 1 0.7"
"RSV 250311 YoungEarly X X 15 1 0.7"
"RSV 250311 YoungEarly nb X 15 1 0.7"
)

combination="${combinations[$SLURM_ARRAY_TASK_ID]}"

ipython Analyses/fit_opt.py $combination