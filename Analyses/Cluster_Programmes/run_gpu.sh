#!/bin/bash
#SBATCH --job-name=optax
#SBATCH --account=ac_idmodels
#SBATCH --partition=savio4_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A5000:1
#SBATCH --qos=a5k_gpu4_normal
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bjsinger@berkeley.edu
#SBATCH --time=24:00:00

# Array job specifications:
#SBATCH --array=0-5
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

module purge
module load anaconda3
source activate /global/scratch/users/bjsinger/jax_env

PATHOGENS=("RSV" "Metapneumovirus" "Adenovirus" "Parainfluenza3" "InfluenzaA" "InfluenzaB")
CURRENT_PATHOGEN=${PATHOGENS[$SLURM_ARRAY_TASK_ID]}
echo "Starting optimization for: $CURRENT_PATHOGEN (Task ID: $SLURM_ARRAY_TASK_ID)"
python -u Analyses/fit_opt.py $CURRENT_PATHOGEN 2603172 Exponential NA flexagep01 1e-9 100 2000 0.5 optax