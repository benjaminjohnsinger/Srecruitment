#!/bin/bash
#SBATCH --job-name=bbDE2
#SBATCH --account=fc_coronamodel
#SBATCH --partition=savio2
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --array=0-11
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bjsinger@berkeley.edu
## Command(s) to run:

module load python/3.11.6-gcc-11.4.0

combinations=(
"RSV 250521 FlexStepwise setimport flexage 0.01 20 1.2 0.7"
"InfluenzaA 250521 FlexStepwise setimport flexage 0.01 20 1.2 0.7"
"InfluenzaB 250521 FlexStepwise setimport flexage 0.01 20 1.2 0.7"
"Metapneumovirus 250521 FlexStepwise setimport flexage 0.01 20 1.2 0.7"
"Adenovirus 250521 FlexStepwise setimport flexage 0.01 20 1.2 0.7"
"Parainfluenza3 250521 FlexStepwise setimport flexage 0.01 20 1.2 0.7"
"RSV 2505212 FlexStepwise setimport flexage 0.01 40 1 0.7"
"InfluenzaA 2505212 FlexStepwise setimport flexage 0.01 40 1 0.7"
"InfluenzaB 2505212 FlexStepwise setimport flexage 0.01 40 1 0.7"
"Metapneumovirus 2505212 FlexStepwise setimport flexage 0.01 40 1 0.7"
"Adenovirus 2505212 FlexStepwise setimport flexage 0.01 40 1 0.7"
"Parainfluenza3 2505212 FlexStepwise setimport flexage 0.01 40 1 0.7"
)

combination="${combinations[$SLURM_ARRAY_TASK_ID]}"

ipython Analyses/fit_opt.py $combination