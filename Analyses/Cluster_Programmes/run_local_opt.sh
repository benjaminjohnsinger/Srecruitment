#!/bin/bash
#SBATCH --job-name=lhnmfluA
#SBATCH --account=
#SBATCH --partition=savio2_htc
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --output=lhnmfluA_%j.out
#SBATCH --error=lhnmfluA_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bjsinger@berkeley.edu
## Command(s) to run:

module load python/3.11.6-gcc-11.4.0
ipython local_opt.py InfluenzaA 24