#!/bin/bash
#SBATCH --job-name=DErsi
#SBATCH --account=ac_idmodels
#SBATCH --partition=savio2
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --array=0-3
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bjsinger@berkeley.edu
## Command(s) to run:

module load python/3.11.6-gcc-11.4.0

combinations=(
"RSV 250317 Mobility X X 15 1 0.7"
"RSV 250317 Mobility nb X 15 1 0.7"
"RSV 250317 FlexStepwise X X 15 1 0.7"
"RSV 250317 FlexStepwise nb X 15 1 0.7"
"InfluenzaA 250317 Mobility X X 15 1 0.7"
"InfluenzaA 250317 Mobility X nr 15 1 0.7"
"InfluenzaA 250317 Mobility nb X 15 1 0.7"
"InfluenzaA 250317 Mobility nb nr 15 1 0.7"
"InfluenzaA 250317 FlexStepwise X X 15 1 0.7"
"InfluenzaA 250317 FlexStepwise X nr 15 1 0.7"
"InfluenzaA 250317 FlexStepwise nb X 15 1 0.7"
"InfluenzaA 250317 FlexStepwise nb nr 15 1 0.7"
"InfluenzaB 250317 Mobility X X 15 1 0.7"
"InfluenzaB 250317 Mobility X nr 15 1 0.7"
"InfluenzaB 250317 Mobility nb X 15 1 0.7"
"InfluenzaB 250317 Mobility nb nr 15 1 0.7"
"InfluenzaB 250317 FlexStepwise X X 15 1 0.7"
"InfluenzaB 250317 FlexStepwise X nr 15 1 0.7"
"InfluenzaB 250317 FlexStepwise nb X 15 1 0.7"
"InfluenzaB 250317 FlexStepwise nb nr 15 1 0.7"
)

combination="${combinations[$SLURM_ARRAY_TASK_ID]}"

ipython Analyses/fit_opt.py $combination