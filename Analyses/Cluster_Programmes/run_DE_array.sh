#!/bin/bash
#SBATCH --job-name=ppnewvx
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
source ~/match-env/bin/activate

combinations=(
"RSV 260217 FlexStepwise NA flexage 1e-9 20 1 0.7"
"RSV 2602172 FlexStepwise NA flexage 1e-9 20 1 0.7"
"RSV 2602173 FlexStepwise NA flexage 1e-9 20 1 0.7"
"InfluenzaA 260217 FlexStepwise NA flexage 1e-9 20 1 0.7"
"InfluenzaA 2602172 FlexStepwise NA flexage 1e-9 20 1 0.7"
"InfluenzaA 2602173 FlexStepwise NA flexage 1e-9 20 1 0.7"
"InfluenzaB 260217 FlexStepwise NA flexage 1e-9 20 1 0.7"
"InfluenzaB 2602172 FlexStepwise NA flexage 1e-9 20 1 0.7"
"InfluenzaB 2602173 FlexStepwise NA flexage 1e-9 20 1 0.7"
"Metapneumovirus 260217 FlexStepwise NA flexage 1e-9 20 1 0.7"
"Metapneumovirus 2602172 FlexStepwise NA flexage 1e-9 20 1 0.7"
"Metapneumovirus 2602173 FlexStepwise NA flexage 1e-9 20 1 0.7"
"Adenovirus 260217 FlexStepwise NA flexage 1e-9 20 1 0.7"
"Adenovirus 2602172 FlexStepwise NA flexage 1e-9 20 1 0.7"
"Adenovirus 2602173 FlexStepwise NA flexage 1e-9 20 1 0.7"
"Parainfluenza3 260217 FlexStepwise NA flexage 1e-9 20 1 0.7"
"Parainfluenza3 2602172 FlexStepwise NA flexage 1e-9 20 1 0.7"
"Parainfluenza3 2602173 FlexStepwise NA flexage 1e-9 20 1 0.7"
)

combination="${combinations[$SLURM_ARRAY_TASK_ID]}"

python -u Analyses/fit_opt.py $combination >> Outputs/DE_Outputs/$(echo "$combination" | tr -d ' ').csv