#!/bin/bash
#SBATCH --job-name=bigDE
#SBATCH --account=fc_coronamodel
#SBATCH --partition=savio2
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --array=0-5
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bjsinger@berkeley.edu
## Command(s) to run:

module load python/3.11.6-gcc-11.4.0

combinations=(
"RSV 250514 FlexStepwise setimport flexage 0.01 20 1.2 0.8"
"InfluenzaA 250514 FlexStepwise setimport flexage 0.01 20 1.2 0.8"
"InfluenzaB 250514 FlexStepwise setimport flexage 0.01 20 1.2 0.8"
"Metapneumovirus 250514 FlexStepwise setimport flexage 0.01 20 1.2 0.8"
"Adenovirus 250514 FlexStepwise setimport flexage 0.01 20 1.2 0.8"
"Parainfluenza3 250514 FlexStepwise setimport flexage 0.01 20 1.2 0.8"
)

combination="${combinations[$SLURM_ARRAY_TASK_ID]}"

ipython Analyses/fit_opt.py $combination