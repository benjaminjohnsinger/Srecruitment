#!/bin/bash
#SBATCH --job-name=long
#SBATCH --account=ac_idmodels
#SBATCH --partition=savio4_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A5000:1
#SBATCH --qos=a5k_gpu4_normal
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bjsinger@berkeley.edu
#SBATCH --time=72:00:00

# Array job specifications:
#SBATCH --array=0-8
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

module purge
module load anaconda3
source activate /global/scratch/users/bjsinger/jax_env

combinations=(
"RSV 260415 ExponentialInOut maxmimmsplit daycarep5maxagep028 1e-9 200 4000 0.7"
"Metapneumovirus 260415 ExponentialInOut maxmimmsplit daycarep5maxagep02 1e-9 200 4000 0.7"
"InfluenzaA 260415 ExponentialInOut maxmimmsplit daycarep5maxagep05 1e-9 200 4000 0.7"
"InfluenzaB 260415 ExponentialInOut maxmimmsplit daycarep5maxagep05 1e-9 200 4000 0.7"
"Adenovirus 260415 ExponentialInOut maxmimmsplit daycarep5maxagep02 1e-9 200 4000 0.7"
"Parainfluenza3 260415 ExponentialInOut maxmimmsplit daycarep5maxagep02 1e-9 200 4000 0.7"
)

combination="${combinations[$SLURM_ARRAY_TASK_ID]}"

python -u Analyses/fit_opt.py $combination