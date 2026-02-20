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
"RSV 260220 FlexStepwise NA 2020-03-19 1e-9 20 1 0.7"
"RSV 2602202 FlexStepwise NA 2020-03-19 1e-9 20 1 0.7"
"RSV 2602203 FlexStepwise NA 2020-03-19 1e-9 20 1 0.7"
"InfluenzaA 260220 FlexStepwise NA 2020-03-19 1e-9 20 1 0.7"
"InfluenzaA 2602202 FlexStepwise NA 2020-03-19 1e-9 20 1 0.7"
"InfluenzaA 2602203 FlexStepwise NA 2020-03-19 1e-9 20 1 0.7"
"InfluenzaB 260220 FlexStepwise NA 2020-03-19 1e-9 20 1 0.7"
"InfluenzaB 2602202 FlexStepwise NA 2020-03-19 1e-9 20 1 0.7"
"InfluenzaB 2602203 FlexStepwise NA 2020-03-19 1e-9 20 1 0.7"
"Metapneumovirus 260220 FlexStepwise NA 2020-03-19 1e-9 20 1 0.7"
"Metapneumovirus 2602202 FlexStepwise NA 2020-03-19 1e-9 20 1 0.7"
"Metapneumovirus 2602203 FlexStepwise NA 2020-03-19 1e-9 20 1 0.7"
"Adenovirus 260220 FlexStepwise NA 2020-03-19 1e-9 20 1 0.7"
"Adenovirus 2602202 FlexStepwise NA 2020-03-19 1e-9 20 1 0.7"
"Adenovirus 2602203 FlexStepwise NA 2020-03-19 1e-9 20 1 0.7"
"Parainfluenza3 260220 FlexStepwise NA 2020-03-19 1e-9 20 1 0.7"
"Parainfluenza3 2602202 FlexStepwise NA 2020-03-19 1e-9 20 1 0.7"
"Parainfluenza3 2602203 FlexStepwise NA 2020-03-19 1e-9 20 1 0.7"
)

combination="${combinations[$SLURM_ARRAY_TASK_ID]}"

python -u Analyses/fit_opt.py $combination >> Outputs/DE_Outputs/$(echo "$combination" | tr -d ' ').csv