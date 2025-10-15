#!/bin/bash
#SBATCH --job-name=mobility
#SBATCH --account=fc_coronamodel
#SBATCH --partition=savio2
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --array=0-25
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bjsinger@berkeley.edu
## Command(s) to run:

module load python/3.11.6-gcc-11.4.0
source ~/match-env/bin/activate

combinations=(
"RSV 251014 Mobility NA flexage 1e-9 20 1 0.7"
"RSV 2510142 Mobility NA flexage 1e-9 20 1 0.7"
"RSV 251014 Mobility2 NA flexage 1e-9 20 1 0.7"
"RSV 2510142 Mobility2 NA flexage 1e-9 20 1 0.7"
"InfluenzaA 251014 Mobility NA flexage 1e-9 20 1 0.7"
"InfluenzaA 2510142 Mobility NA flexage 1e-9 20 1 0.7"
"InfluenzaA 251014 Mobility2 NA flexage 1e-9 20 1 0.7"
"InfluenzaA 2510142 Mobility2 NA flexage 1e-9 20 1 0.7"
"InfluenzaB 251014 Mobility NA flexage 1e-9 20 1 0.7"
"InfluenzaB 2510142 Mobility NA flexage 1e-9 20 1 0.7"
"InfluenzaB 251014 Mobility2 NA flexage 1e-9 20 1 0.7"
"InfluenzaB 2510142 Mobility2 NA flexage 1e-9 20 1 0.7"
"Metapneumovirus 251014 Mobility NA flexage 1e-9 20 1 0.7"
"Metapneumovirus 2510142 Mobility NA flexage 1e-9 20 1 0.7"
"Metapneumovirus 251014 Mobility2 NA flexage 1e-9 20 1 0.7"
"Metapneumovirus 2510142 Mobility2 NA flexage 1e-9 20 1 0.7"
"Adenovirus 251014 Mobility NA flexage 1e-9 20 1 0.7"
"Adenovirus 2510142 Mobility NA flexage 1e-9 20 1 0.7"
"Adenovirus 251014 Mobility2 NA flexage 1e-9 20 1 0.7"
"Adenovirus 2510142 Mobility2 NA flexage 1e-9 20 1 0.7"
"Parainfluenza3 251014 Mobility NA flexage 1e-9 20 1 0.7"
"Parainfluenza3 2510142 Mobility NA flexage 1e-9 20 1 0.7"
"Parainfluenza3 251014 Mobility2 NA flexage 1e-9 20 1 0.7"
"Parainfluenza3 2510142 Mobility2 NA flexage 1e-9 20 1 0.7"
"test 251014 FlexStepwise NA flexage 1e-9 20 1 0.7"
"test 2510142 FlexStepwise NA flexage 1e-9 20 1 0.7"
)

combination="${combinations[$SLURM_ARRAY_TASK_ID]}"

python -u Analyses/fit_opt.py $combination >> Outputs/DE_Outputs/$(echo "$combination" | tr -d ' ').csv