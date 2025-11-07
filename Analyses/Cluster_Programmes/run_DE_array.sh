#!/bin/bash
#SBATCH --job-name=wane
#SBATCH --account=fc_coronamodel
#SBATCH --partition=savio2
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --array=0-13
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bjsinger@berkeley.edu
## Command(s) to run:

module load python/3.11.6-gcc-11.4.0
source ~/match-env/bin/activate

combinations=(
"InfluenzaB 251106 FlexStepwise wane flexage 1e-9 20 1 0.7"
"InfluenzaB 2511062 FlexStepwise wane flexage 1e-9 20 1 0.7"
"InfluenzaB 2511063 FlexStepwise wane flexage 1e-9 20 1 0.7"
"InfluenzaB 2511064 FlexStepwise wane flexage 1e-9 20 1 0.7"
"InfluenzaB 2511065 FlexStepwise wane flexage 1e-9 20 1 0.7"
"InfluenzaB 2511066 FlexStepwise wane flexage 1e-9 20 1 0.7"
"InfluenzaB 2511067 FlexStepwise wane flexage 1e-9 20 1 0.7"
"InfluenzaB 251106 FlexStepwise maxmimmwane flexage 1e-9 20 1 0.7"
"InfluenzaB 2511062 FlexStepwise maxmimmwane flexage 1e-9 20 1 0.7"
"InfluenzaB 2511063 FlexStepwise maxmimmwane flexage 1e-9 20 1 0.7"
"InfluenzaB 2511064 FlexStepwise maxmimmwane flexage 1e-9 20 1 0.7"
"InfluenzaB 2511065 FlexStepwise maxmimmwane flexage 1e-9 20 1 0.7"
"InfluenzaB 2511066 FlexStepwise maxmimmwane flexage 1e-9 20 1 0.7"
"InfluenzaB 2511067 FlexStepwise maxmimmwane flexage 1e-9 20 1 0.7"
)

combination="${combinations[$SLURM_ARRAY_TASK_ID]}"

python -u Analyses/fit_opt.py $combination >> Outputs/DE_Outputs/$(echo "$combination" | tr -d ' ').csv