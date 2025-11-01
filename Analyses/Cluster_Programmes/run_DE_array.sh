#!/bin/bash
#SBATCH --job-name=prevaxDE
#SBATCH --account=fc_coronamodel
#SBATCH --partition=savio2
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --array=0-79
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bjsinger@berkeley.edu
## Command(s) to run:

module load python/3.11.6-gcc-11.4.0
source ~/match-env/bin/activate

combinations=(
"0test0 251031 FlexStepwise NA flexage 1e-9 20 1 0.7"
"0test1 251031 FlexStepwise NA flexage 1e-9 20 1 0.7"
"1test0 251031 FlexStepwise NA flexage 1e-9 20 1 0.7"
"1test1 251031 FlexStepwise NA flexage 1e-9 20 1 0.7"
"2test0 251031 FlexStepwise NA flexage 1e-9 20 1 0.7"
"2test1 251031 FlexStepwise NA flexage 1e-9 20 1 0.7"
"3test0 251031 FlexStepwise NA flexage 1e-9 20 1 0.7"
"3test1 251031 FlexStepwise NA flexage 1e-9 20 1 0.7"
"4test0 251031 FlexStepwise NA flexage 1e-9 20 1 0.7"
"4test1 251031 FlexStepwise NA flexage 1e-9 20 1 0.7"
"5test0 251031 FlexStepwise NA flexage 1e-9 20 1 0.7"
"5test1 251031 FlexStepwise NA flexage 1e-9 20 1 0.7"
"0test0 2510312 FlexStepwise NA flexage 1e-9 20 1 0.7"
"0test1 2510312 FlexStepwise NA flexage 1e-9 20 1 0.7"
"1test0 2510312 FlexStepwise NA flexage 1e-9 20 1 0.7"
"1test1 2510312 FlexStepwise NA flexage 1e-9 20 1 0.7"
"2test0 2510312 FlexStepwise NA flexage 1e-9 20 1 0.7"
"2test1 2510312 FlexStepwise NA flexage 1e-9 20 1 0.7"
"3test0 2510312 FlexStepwise NA flexage 1e-9 20 1 0.7"
"3test1 2510312 FlexStepwise NA flexage 1e-9 20 1 0.7"
"4test0 2510312 FlexStepwise NA flexage 1e-9 20 1 0.7"
"4test1 2510312 FlexStepwise NA flexage 1e-9 20 1 0.7"
"5test0 2510312 FlexStepwise NA flexage 1e-9 20 1 0.7"
"5test1 2510312 FlexStepwise NA flexage 1e-9 20 1 0.7"
"0test2 251031 FlexStepwise NA flexage 1e-9 20 1 0.7"
"0test3 251031 FlexStepwise NA flexage 1e-9 20 1 0.7"
"1test2 251031 FlexStepwise NA flexage 1e-9 20 1 0.7"
"1test3 251031 FlexStepwise NA flexage 1e-9 20 1 0.7"
"2test2 251031 FlexStepwise NA flexage 1e-9 20 1 0.7"
"2test3 251031 FlexStepwise NA flexage 1e-9 20 1 0.7"
"3test2 251031 FlexStepwise NA flexage 1e-9 20 1 0.7"
"3test3 251031 FlexStepwise NA flexage 1e-9 20 1 0.7"
"4test2 251031 FlexStepwise NA flexage 1e-9 20 1 0.7"
"4test3 251031 FlexStepwise NA flexage 1e-9 20 1 0.7"
"5test2 251031 FlexStepwise NA flexage 1e-9 20 1 0.7"
"5test3 251031 FlexStepwise NA flexage 1e-9 20 1 0.7"
"0test2 2510312 FlexStepwise NA flexage 1e-9 20 1 0.7"
"0test3 2510312 FlexStepwise NA flexage 1e-9 20 1 0.7"
"1test2 2510312 FlexStepwise NA flexage 1e-9 20 1 0.7"
"1test3 2510312 FlexStepwise NA flexage 1e-9 20 1 0.7"
"2test2 2510312 FlexStepwise NA flexage 1e-9 20 1 0.7"
"2test3 2510312 FlexStepwise NA flexage 1e-9 20 1 0.7"
"3test2 2510312 FlexStepwise NA flexage 1e-9 20 1 0.7"
"3test3 2510312 FlexStepwise NA flexage 1e-9 20 1 0.7"
"4test2 2510312 FlexStepwise NA flexage 1e-9 20 1 0.7"
"4test3 2510312 FlexStepwise NA flexage 1e-9 20 1 0.7"
"5test2 2510312 FlexStepwise NA flexage 1e-9 20 1 0.7"
"5test3 2510312 FlexStepwise NA flexage 1e-9 20 1 0.7"
"0test2 2510313 FlexStepwise NA flexage 1e-9 20 1 0.7"
"0test3 2510313 FlexStepwise NA flexage 1e-9 20 1 0.7"
"1test2 2510313 FlexStepwise NA flexage 1e-9 20 1 0.7"
"1test3 2510313 FlexStepwise NA flexage 1e-9 20 1 0.7"
"2test2 2510313 FlexStepwise NA flexage 1e-9 20 1 0.7"
"2test3 2510313 FlexStepwise NA flexage 1e-9 20 1 0.7"
"3test2 2510313 FlexStepwise NA flexage 1e-9 20 1 0.7"
"3test3 2510313 FlexStepwise NA flexage 1e-9 20 1 0.7"
"4test2 2510313 FlexStepwise NA flexage 1e-9 20 1 0.7"
"4test3 2510313 FlexStepwise NA flexage 1e-9 20 1 0.7"
"5test2 2510313 FlexStepwise NA flexage 1e-9 20 1 0.7"
"5test3 2510313 FlexStepwise NA flexage 1e-9 20 1 0.7"
"0test2 2510314 FlexStepwise NA flexage 1e-9 20 1 0.7"
"0test3 2510314 FlexStepwise NA flexage 1e-9 20 1 0.7"
"1test2 2510314 FlexStepwise NA flexage 1e-9 20 1 0.7"
"1test3 2510314 FlexStepwise NA flexage 1e-9 20 1 0.7"
"2test2 2510314 FlexStepwise NA flexage 1e-9 20 1 0.7"
"2test3 2510314 FlexStepwise NA flexage 1e-9 20 1 0.7"
"3test2 2510314 FlexStepwise NA flexage 1e-9 20 1 0.7"
"3test3 2510314 FlexStepwise NA flexage 1e-9 20 1 0.7"
"4test2 2510314 FlexStepwise NA flexage 1e-9 20 1 0.7"
"4test3 2510314 FlexStepwise NA flexage 1e-9 20 1 0.7"
"5test2 2510314 FlexStepwise NA flexage 1e-9 20 1 0.7"
"5test3 2510314 FlexStepwise NA flexage 1e-9 20 1 0.7"
)

combination="${combinations[$SLURM_ARRAY_TASK_ID]}"

python -u Analyses/fit_opt.py $combination >> Outputs/DE_Outputs/$(echo "$combination" | tr -d ' ').csv