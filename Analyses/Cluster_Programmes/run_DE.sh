#!/bin/bash
#SBATCH --job-name=testdata
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
# ipython Analyses/cm_opt.py 250501 15 1 0.7
ipython Analyses/fit_opt.py test 251013 FlexStepwise NA flexage 1e-9 20 1 0.7