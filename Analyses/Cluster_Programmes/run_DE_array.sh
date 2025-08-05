#!/bin/bash
#SBATCH --job-name=seasinf
#SBATCH --account=fc_coronamodel
#SBATCH --partition=savio2
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --array=0-17
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bjsinger@berkeley.edu
## Command(s) to run:

module load python/3.11.6-gcc-11.4.0

combinations=(
"RSV 250805 FlexStepwise season_infections flexage 0.01 20 1 0.7 0.01"
"InfluenzaA 250805 FlexStepwise season_infections flexage 0.01 20 1 0.7 0.01"
"InfluenzaB 250805 FlexStepwise season_infections flexage 0.01 20 1 0.7 0.01"
"Metapneumovirus 250805 FlexStepwise season_infections flexage 0.01 20 1 0.7 0.01"
"Adenovirus 250805 FlexStepwise season_infections flexage 0.01 20 1 0.7 0.01"
"Parainfluenza3 250805 FlexStepwise season_infections flexage 0.01 20 1 0.7 0.01"
"RSV 2508052 FlexStepwise season_infections flexage 0.01 20 1 0.7 0.01"
"InfluenzaA 2508052 FlexStepwise season_infections flexage 0.01 20 1 0.7 0.01"
"InfluenzaB 2508052 FlexStepwise season_infections flexage 0.01 20 1 0.7 0.01"
"Metapneumovirus 2508052 FlexStepwise season_infections flexage 0.01 20 1 0.7 0.01"
"Adenovirus 2508052 FlexStepwise season_infections flexage 0.01 20 1 0.7 0.01"
"Parainfluenza3 2508052 FlexStepwise season_infections flexage 0.01 20 1 0.7 0.01"
"RSV 2508053 FlexStepwise season_infections flexage 0.01 20 1 0.7 0.01"
"InfluenzaA 2508053 FlexStepwise season_infections flexage 0.01 20 1 0.7 0.01"
"InfluenzaB 2508053 FlexStepwise season_infections flexage 0.01 20 1 0.7 0.01"
"Metapneumovirus 2508053 FlexStepwise season_infections flexage 0.01 20 1 0.7 0.01"
"Adenovirus 2508053 FlexStepwise season_infections flexage 0.01 20 1 0.7 0.01"
"Parainfluenza3 2508053 FlexStepwise season_infections flexage 0.01 20 1 0.7 0.01"
)

combination="${combinations[$SLURM_ARRAY_TASK_ID]}"

ipython Analyses/fit_opt.py $combination