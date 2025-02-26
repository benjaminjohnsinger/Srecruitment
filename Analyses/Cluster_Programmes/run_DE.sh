#!/bin/bash
#SBATCH --job-name=deRSVS2D2
#SBATCH --account=ac_idmodels
#SBATCH --partition=savio2
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bjsinger@berkeley.edu
## Command(s) to run:

module load python/3.11.6-gcc-11.4.0
ipython Analyses/fit_opt.py RSV 250226 15 1 0.7
