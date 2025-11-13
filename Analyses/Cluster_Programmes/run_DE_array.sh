#!/bin/bash
#SBATCH --job-name=freeflu
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
"RSV 251112 maximmwaneflexage2511032 maxmimmwane flexage 1e-9 20 1 0.7"
"RSV 2511122 maximmwaneflexage2511032 maxmimmwane flexage 1e-9 20 1 0.7"
"Metpneumovirus 251112 maximmwaneflexage2511032 maxmimmwane flexage 1e-9 20 1 0.7"
"Metpneumovirus 2511122 maximmwaneflexage2511032 maxmimmwane flexage 1e-9 20 1 0.7"
"InfluenzaA 251112 maximmwaneflexage2511032 maxmimmwane flexage 1e-9 20 1 0.7"
"InfluenzaA 2511122 maximmwaneflexage2511032 maxmimmwane flexage 1e-9 20 1 0.7"
"InfluenzaB 251112 maximmwaneflexage2511032 maxmimmwane flexage 1e-9 20 1 0.7"
"InfluenzaB 2511122 maximmwaneflexage2511032 maxmimmwane flexage 1e-9 20 1 0.7"
"InfluenzaB 2511123 maximmwaneflexage2511032 maxmimmwane flexage 1e-9 20 1 0.7"
"InfluenzaB 2511124 maximmwaneflexage2511032 maxmimmwane flexage 1e-9 20 1 0.7"
"Adenovirus 251112 maximmwaneflexage2511032 maxmimmwane flexage 1e-9 20 1 0.7"
"Adenovirus 2511122 maximmwaneflexage2511032 maxmimmwane flexage 1e-9 20 1 0.7"
"Parainfluenza3 251112 maximmwaneflexage2511032 maxmimmwane flexage 1e-9 20 1 0.7"
"Parainfluenza3 2511122 maximmwaneflexage2511032 maxmimmwane flexage 1e-9 20 1 0.7"
)

combination="${combinations[$SLURM_ARRAY_TASK_ID]}"

python -u Analyses/fit_opt.py $combination >> Outputs/DE_Outputs/$(echo "$combination" | tr -d ' ').csv