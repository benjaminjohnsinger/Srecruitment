#!/bin/bash
#SBATCH --job-name=tol1e-4
#SBATCH --account=fc_coronamodel
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
"InfluenzaA 250805701701 FlexStepwise season_infections flexage 0.01 20 1 0.7 0.0001"
"Metapneumovirus 250805701701 FlexStepwise season_infections flexage 0.01 20 1 0.7 0.0001"
"InfluenzaA 2508057017012 FlexStepwise season_infections flexage 0.01 20 1 0.7 0.0001"
"Metapneumovirus 2508057017012 FlexStepwise season_infections flexage 0.01 20 1 0.7 0.0001"
)

combination="${combinations[$SLURM_ARRAY_TASK_ID]}"

ipython Analyses/fit_opt.py $combination