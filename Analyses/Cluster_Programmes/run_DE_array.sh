#!/bin/bash
#SBATCH --job-name=mobility
#SBATCH --account=fc_coronamodel
#SBATCH --partition=savio2
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --array=0-35
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bjsinger@berkeley.edu
## Command(s) to run:

module load python/3.11.6-gcc-11.4.0
source ~/match-env/bin/activate

combinations=(
"RSV 250908 Taube NA flexage 1e-9 20 1 0.7"
"Metapneumovirus 250908 Taube NA flexage 1e-9 20 1 0.7"
"Adenovirus 250908 Taube NA flexage 1e-9 20 1 0.7"
"Parainfluenza3 250908 Taube NA flexage 1e-9 20 1 0.7"
"InfluenzaA 250908 Taube NA flexage 1e-9 20 1 0.7"
"InfluenzaB 250908 Taube NA flexage 1e-9 20 1 0.7"
"RSV 2509082 Taube NA flexage 1e-9 20 1 0.7"
"Metapneumovirus 2509082 Taube NA flexage 1e-9 20 1 0.7"
"Adenovirus 2509082 Taube NA flexage 1e-9 20 1 0.7"
"Parainfluenza3 2509082 Taube NA flexage 1e-9 20 1 0.7"
"InfluenzaA 2509082 Taube NA flexage 1e-9 20 1 0.7"
"InfluenzaB 2509082 Taube NA flexage 1e-9 20 1 0.7"
"RSV 250908 Mobility NA flexage 1e-9 20 1 0.7"
"Metapneumovirus 250908 Mobility NA flexage 1e-9 20 1 0.7"
"Adenovirus 250908 Mobility NA flexage 1e-9 20 1 0.7"
"Parainfluenza3 250908 Mobility NA flexage 1e-9 20 1 0.7"
"InfluenzaA 250908 Mobility NA flexage 1e-9 20 1 0.7"
"InfluenzaB 250908 Mobility NA flexage 1e-9 20 1 0.7"
"RSV 2509082 Mobility NA flexage 1e-9 20 1 0.7"
"Metapneumovirus 2509082 Mobility NA flexage 1e-9 20 1 0.7"
"Adenovirus 2509082 Mobility NA flexage 1e-9 20 1 0.7"
"Parainfluenza3 2509082 Mobility NA flexage 1e-9 20 1 0.7"
"InfluenzaA 2509082 Mobility NA flexage 1e-9 20 1 0.7"
"InfluenzaB 2509082 Mobility NA flexage 1e-9 20 1 0.7"
"RSV 250908 Mobility2 NA flexage 1e-9 20 1 0.7"
"Metapneumovirus 250908 Mobility2 NA flexage 1e-9 20 1 0.7"
"Adenovirus 250908 Mobility2 NA flexage 1e-9 20 1 0.7"
"Parainfluenza3 250908 Mobility2 NA flexage 1e-9 20 1 0.7"
"InfluenzaA 250908 Mobility2 NA flexage 1e-9 20 1 0.7"
"InfluenzaB 250908 Mobility2 NA flexage 1e-9 20 1 0.7"
"RSV 2509082 Mobility2 NA flexage 1e-9 20 1 0.7"
"Metapneumovirus 2509082 Mobility2 NA flexage 1e-9 20 1 0.7"
"Adenovirus 2509082 Mobility2 NA flexage 1e-9 20 1 0.7"
"Parainfluenza3 2509082 Mobility2 NA flexage 1e-9 20 1 0.7"
"InfluenzaA 2509082 Mobility2 NA flexage 1e-9 20 1 0.7"
"InfluenzaB 2509082 Mobility2 NA flexage 1e-9 20 1 0.7"
)

combination="${combinations[$SLURM_ARRAY_TASK_ID]}"

ipython Analyses/fit_opt.py $combination