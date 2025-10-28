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
ipython Analyses/cm_opt.py 0test0,1test0,2test0,3test0,4test0,5test0 251024 NA flexage 20 1 0.7 >> Outputs/CM_Outputs/test0_251024.csv
ipython Analyses/cm_opt.py 0test1,1test1,2test1,3test1,4test1,5test1 251024 NA flexage 20 1 0.7 >> Outputs/CM_Outputs/test1_251024.csv
ipython Analyses/cm_opt.py 0test0,1test0,2test0,3test0,4test0,5test0 2510242 NA flexage 20 1 0.7 >> Outputs/CM_Outputs/test0_2510242.csv