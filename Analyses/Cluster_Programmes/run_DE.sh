#!/bin/bash
#SBATCH --job-name=cm
#SBATCH --account=fc_coronamodel
#SBATCH --partition=savio2
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bjsinger@berkeley.edu
## Command(s) to run:

module load python/3.11.6-gcc-11.4.0
ipython Analyses/cm_opt.py RSV,Metapneumovirus,InfluenzaA,InfluenzaB,Adenovirus,Parainfluenza3 2511032,2511032,2511032,241104,2511032,2511032 maxmimmwane,maxmimmwane,maxmimmwane,wane,maxmimmwane,maxmimmwane flexage 20 1 0.7 >> Outputs/CM_Outputs/251104.csv