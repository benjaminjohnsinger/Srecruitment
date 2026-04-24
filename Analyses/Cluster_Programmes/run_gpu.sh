#!/bin/bash
#SBATCH --job-name=ODip+
#SBATCH --account=ac_idmodels
#SBATCH --partition=savio4_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A5000:1
#SBATCH --qos=a5k_gpu4_normal
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bjsinger@berkeley.edu
#SBATCH --time=10:00:00

# Array job specifications:
#SBATCH --array=0-6
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

module purge
module load anaconda3
source activate /global/scratch/users/bjsinger/jax_env

combinations=(
"RSV 260423 ExponentialODipEqual split maxagep028 1e-9 200 2000 0.7"
"Metapneumovirus 260423 ExponentialODipEqual split maxagep028 1e-9 200 2000 0.7"
"InfluenzaA 260423 ExponentialODipEqual split nrmaxagep05 1e-9 200 2000 0.7"
"InfluenzaB 260423 ExponentialODipEqual split nrmaxagep05 1e-9 200 2000 0.7"
"InfluenzaA 260424 ExponentialODipEqual split nrmaxagep05 1e-9 200 2000 0.7"
"InfluenzaB 260424 ExponentialODipEqual split nrmaxagep05 1e-9 200 2000 0.7"
"Metapneumovirus 260424 ExponentialODipEqual split maxagep028 1e-9 200 2000 0.7"
)

combination="${combinations[$SLURM_ARRAY_TASK_ID]}"

python -u Analyses/fit_opt.py $combination