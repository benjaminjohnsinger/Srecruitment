#!/bin/bash
#SBATCH --job-name=testDE
#SBATCH --account=fc_coronamodel
#SBATCH --partition=savio2
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --array=0-11
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bjsinger@berkeley.edu
## Command(s) to run:

module load python/3.11.6-gcc-11.4.0
source ~/match-env/bin/activate

combinations=(
"0test0 251024 FlexStepwise NA flexage 1e-9 20 1 0.7"
"1test0 251024 FlexStepwise NA flexage 1e-9 20 1 0.7"
"2test0 251024 FlexStepwise NA flexage 1e-9 20 1 0.7"
"3test0 251024 FlexStepwise NA flexage 1e-9 20 1 0.7"
"4test0 251024 FlexStepwise NA flexage 1e-9 20 1 0.7"
"5test0 251024 FlexStepwise NA flexage 1e-9 20 1 0.7"
"0test1 251024 FlexStepwise NA flexage 1e-9 20 1 0.7"
"1test1 251024 FlexStepwise NA flexage 1e-9 20 1 0.7"
"2test1 251024 FlexStepwise NA flexage 1e-9 20 1 0.7"
"3test1 251024 FlexStepwise NA flexage 1e-9 20 1 0.7"
"4test1 251024 FlexStepwise NA flexage 1e-9 20 1 0.7"
"5test1 251024 FlexStepwise NA flexage 1e-9 20 1 0.7"
"0test0 2510242 FlexStepwise NA flexage 1e-9 20 1 0.7"
"1test0 2510242 FlexStepwise NA flexage 1e-9 20 1 0.7"
"2test0 2510242 FlexStepwise NA flexage 1e-9 20 1 0.7"
"3test0 2510242 FlexStepwise NA flexage 1e-9 20 1 0.7"
"4test0 2510242 FlexStepwise NA flexage 1e-9 20 1 0.7"
"5test0 2510242 FlexStepwise NA flexage 1e-9 20 1 0.7"
)

combination="${combinations[$SLURM_ARRAY_TASK_ID]}"

python -u Analyses/fit_opt.py $combination >> Outputs/DE_Outputs/$(echo "$combination" | tr -d ' ').csv