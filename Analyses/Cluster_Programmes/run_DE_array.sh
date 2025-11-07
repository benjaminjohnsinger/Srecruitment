#!/bin/bash
#SBATCH --job-name=fluBwane
#SBATCH --account=fc_coronamodel
#SBATCH --partition=savio2
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --array=0-15
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bjsinger@berkeley.edu
## Command(s) to run:

module load python/3.11.6-gcc-11.4.0
source ~/match-env/bin/activate

combinations=(
"InfluenzaB 251107 FlexStepwise wane flexage 1e-9 20 1 0.7"
"InfluenzaB 2511072 FlexStepwise wane flexage 1e-9 20 1 0.7"
"InfluenzaB 2511073 FlexStepwise wane flexage 1e-9 20 1 0.7"
"InfluenzaB 2511074 FlexStepwise wane flexage 1e-9 20 1 0.7"
"InfluenzaB 2511075 FlexStepwise wane flexage 1e-9 20 1 0.7"
"InfluenzaB 2511076 FlexStepwise wane flexage 1e-9 20 1 0.7"
"InfluenzaB 2511077 FlexStepwise wane flexage 1e-9 20 1 0.7"
"InfluenzaB 251107 FlexStepwise maxmimmwane flexage 1e-9 20 1 0.7"
"InfluenzaB 2511072 FlexStepwise maxmimmwane flexage 1e-9 20 1 0.7"
"InfluenzaB 2511073 FlexStepwise maxmimmwane flexage 1e-9 20 1 0.7"
"InfluenzaB 2511074 FlexStepwise maxmimmwane flexage 1e-9 20 1 0.7"
"InfluenzaB 2511075 FlexStepwise maxmimmwane flexage 1e-9 20 1 0.7"
"InfluenzaB 2511076 FlexStepwise maxmimmwane flexage 1e-9 20 1 0.7"
"InfluenzaB 2511077 FlexStepwise maxmimmwane flexage 1e-9 20 1 0.7"
)

combination="${combinations[$SLURM_ARRAY_TASK_ID]}"

python -u Analyses/cm_opt.py $combination >> Outputs/CM_Outputs/$(echo "$combination" | tr -d ' ').csv