#!/bin/bash
#SBATCH --job-name=optax
#SBATCH --account=ac_idmodels
#SBATCH --partition=savio4_gpu
#SBATCH --nodes=1
#
# Number of tasks (one for each GPU desired for use case) (example):
#SBATCH --ntasks=1
#
# Processors per task:
# Four times the number of GPUs for A500 in savio4_gpu
#SBATCH --cpus-per-task=4
#
#Number and type of GPUs
#SBATCH --gres=gpu:A5000:1
#SBATCH --qos=a5k_gpu4_normal

#SBATCH --output=%x_%A.out
#SBATCH --error=%x_%A.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bjsinger@berkeley.edu

# Wall clock limit:
#SBATCH --time=24:00:00

module purge
module load anaconda3
source activate /global/scratch/users/bjsinger/jax_env

python -u Analyses/fit_opt.py RSV 260306 FlexStepwise NA flexagep01 1e-9 500 2000 0.7