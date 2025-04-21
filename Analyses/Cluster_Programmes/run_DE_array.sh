#!/bin/bash
#SBATCH --job-name=oip
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
"RSV 250421 FlexStepwise setimport flexage 0.01 15 1 0.7"
"InfluenzaA 250421 FlexStepwise setimport flexage 0.01 15 1 0.7"
"InfluenzaB 250421 FlexStepwise setimport flexage 0.01 15 1 0.7"
"Metapneumovirus 250421 FlexStepwise setimport flexage 0.01 15 1 0.7"
"Adenovirus 250421 FlexStepwise setimport flexage 0.01 15 1 0.7"
"Parainfluenza3 250421 FlexStepwise setimport flexage 0.01 15 1 0.7"
"Metapneumovirus 2504212 FlexStepwise setimport flexage 0.001 15 1 0.7"
"Adenovirus 2504212 FlexStepwise setimport flexage 0.001 15 1 0.7"
"Parainfluenza3 2504212 FlexStepwise setimport flexage 0.001 15 1 0.7"
)

combination="${combinations[$SLURM_ARRAY_TASK_ID]}"

ipython Analyses/fit_opt.py $combination