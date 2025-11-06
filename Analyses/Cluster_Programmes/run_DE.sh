#!/bin/bash
#SBATCH --job-name=nofluBcmfree
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
ipython Analyses/cm_opt.py RSVfree,Metapneumovirus,InfluenzaAfree,Adenovirus,Parainfluenza3 2511042,2511042,2511042,2511042,2511042 NA,NA,NA,NA,NA flexage 20 1 0.7 >> Outputs/CM_Outputs/2511042noBfree.csv