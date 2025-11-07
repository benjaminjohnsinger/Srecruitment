#!/bin/bash
#SBATCH --job-name=multiNACM
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
"RSV,Metapneumovirus,InfluenzaA,InfluenzaB,Adenovirus,Parainfluenza3 2511032,2511032,251103,2511042,2511032,2511032 NA,NA,NA,NA,NA,NA flexage 20 1 0.7"
"RSV,Metapneumovirus,InfluenzaA,InfluenzaB,Adenovirus,Parainfluenza3 251103,2511032,251103,2511042,2511032,2511032 NA,NA,NA,NA,NA,NA flexage 20 1 0.7"
"RSV,Metapneumovirus,InfluenzaA,InfluenzaB,Adenovirus,Parainfluenza3 2511032,251103,251103,2511042,2511032,2511032 NA,NA,NA,NA,NA,NA flexage 20 1 0.7"
"RSV,Metapneumovirus,InfluenzaA,InfluenzaB,Adenovirus,Parainfluenza3 2511032,2511032,251103,2511042,251103,2511032 NA,NA,NA,NA,NA,NA flexage 20 1 0.7"
"RSV,Metapneumovirus,InfluenzaA,InfluenzaB,Adenovirus,Parainfluenza3 2511032,2511032,251103,2511042,2511032,251103 NA,NA,NA,NA,NA,NA flexage 20 1 0.7"
"RSV,Metapneumovirus,InfluenzaA,InfluenzaB,Adenovirus,Parainfluenza3 251103,251103,251103,2511042,2511032,2511032 NA,NA,NA,NA,NA,NA flexage 20 1 0.7"
"RSV,Metapneumovirus,InfluenzaA,InfluenzaB,Adenovirus,Parainfluenza3 251103,2511032,251103,2511042,251103,2511032 NA,NA,NA,NA,NA,NA flexage 20 1 0.7"
"RSV,Metapneumovirus,InfluenzaA,InfluenzaB,Adenovirus,Parainfluenza3 251103,2511032,251103,2511042,2511032,251103 NA,NA,NA,NA,NA,NA flexage 20 1 0.7"
"RSV,Metapneumovirus,InfluenzaA,InfluenzaB,Adenovirus,Parainfluenza3 2511032,251103,251103,2511042,251103,2511032 NA,NA,NA,NA,NA,NA flexage 20 1 0.7"
"RSV,Metapneumovirus,InfluenzaA,InfluenzaB,Adenovirus,Parainfluenza3 2511032,251103,251103,2511042,2511032,251103 NA,NA,NA,NA,NA,NA flexage 20 1 0.7"
"RSV,Metapneumovirus,InfluenzaA,InfluenzaB,Adenovirus,Parainfluenza3 2511032,2511032,251103,2511042,251103,251103 NA,NA,NA,NA,NA,NA flexage 20 1 0.7"
"RSV,Metapneumovirus,InfluenzaA,InfluenzaB,Adenovirus,Parainfluenza3 251103,251103,251103,2511042,251103,2511032 NA,NA,NA,NA,NA,NA flexage 20 1 0.7"
"RSV,Metapneumovirus,InfluenzaA,InfluenzaB,Adenovirus,Parainfluenza3 251103,251103,251103,2511042,2511032,251103 NA,NA,NA,NA,NA,NA flexage 20 1 0.7"
"RSV,Metapneumovirus,InfluenzaA,InfluenzaB,Adenovirus,Parainfluenza3 251103,2511032,251103,2511042,251103,251103 NA,NA,NA,NA,NA,NA flexage 20 1 0.7"
"RSV,Metapneumovirus,InfluenzaA,InfluenzaB,Adenovirus,Parainfluenza3 2511032,251103,251103,2511042,251103,251103 NA,NA,NA,NA,NA,NA flexage 20 1 0.7"
"RSV,Metapneumovirus,InfluenzaA,InfluenzaB,Adenovirus,Parainfluenza3 251103,251103,251103,2511042,251103,251103 NA,NA,NA,NA,NA,NA flexage 20 1 0.7"
)

combination="${combinations[$SLURM_ARRAY_TASK_ID]}"

python -u Analyses/cm_opt.py $combination >> Outputs/CM_Outputs/$(echo "$combination" | tr -d ' ').csv