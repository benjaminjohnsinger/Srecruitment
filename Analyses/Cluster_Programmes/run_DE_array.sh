#!/bin/bash
#SBATCH --job-name=DEwCMmf032
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
source ~/match-env/bin/activate

combinations=(
"Metapneumovirus 251116 maximmwaneflexage2511032 maxmimmwane flexage 1e-9 20 1 0.7"
"Metapneumovirus 2511162 maximmwaneflexage2511032 maxmimmwane flexage 1e-9 20 1 0.7"
"Metapneumovirus 2511163 maximmwaneflexage2511032 maxmimmwane flexage 1e-9 20 1 0.7"
"Metapneumovirus 2511164 maximmwaneflexage2511032 maxmimmwane flexage 1e-9 20 1 0.7"
"Metapneumovirus 2511165 maximmwaneflexage2511032 maxmimmwane flexage 1e-9 20 1 0.7"
"Metapneumovirus 2511166 maximmwaneflexage2511032 maxmimmwane flexage 1e-9 20 1 0.7"
)

combination="${combinations[$SLURM_ARRAY_TASK_ID]}"

python -u Analyses/fit_opt.py $combination >> Outputs/DE_Outputs/$(echo "$combination" | tr -d ' ').csv