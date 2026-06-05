#!/bin/bash
#SBATCH --job-name=kirlnimm
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
#SBATCH --array=0-31
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

module purge
module load anaconda3
source activate /global/scratch/users/bjsinger/jax_env

combinations=(
"RSV 260605 ExponentialODipLinear kireldedupsac maxagep028 1e-9 200 2000 0.7"
"Metapneumovirus 260605 ExponentialODipLinear kireldedupsac maxagep015 1e-9 200 2000 0.7"
"Adenovirus 260605 ExponentialODipLinear kireldedupsac maxagep003 1e-9 200 2000 0.7"
"Parainfluenza3 260605 ExponentialODipLinear kireldedupsac maxagep004 1e-9 200 2000 0.7"
"InfluenzaA 260605 ExponentialODipLinear kireldedupsac maxagep035 1e-9 200 2000 0.7"
"InfluenzaB 260605 ExponentialODipLinear kireldedupsac maxagep035 1e-9 200 2000 0.7"

"RSV 260605 ExponentialODipLinear unimmlimdedupsac maxagep028 1e-9 200 2000 0.7"
"Metapneumovirus 260605 ExponentialODipLinear unimmlimdedupsac maxagep015 1e-9 200 2000 0.7"
"Adenovirus 260605 ExponentialODipLinear unimmlimdedupsac maxagep003 1e-9 200 2000 0.7"
"Parainfluenza3 260605 ExponentialODipLinear unimmlimdedupsac maxagep004 1e-9 200 2000 0.7"
"InfluenzaA 260605 ExponentialODipLinear unimmlimdedupsac maxagep035 1e-9 200 2000 0.7"
"InfluenzaB 260605 ExponentialODipLinear unimmlimdedupsac maxagep035 1e-9 200 2000 0.7"

"Metapneumovirus 260605 ExponentialODipLinear kireldedupsac maxagep01 1e-9 200 2000 0.7"
"Adenovirus 260605 ExponentialODipLinear kireldedupsac maxagep002 1e-9 200 2000 0.7"
"Parainfluenza3 260605 ExponentialODipLinear kireldedupsac maxagep003 1e-9 200 2000 0.7"
"InfluenzaA 260605 ExponentialODipLinear kireldedupsac maxagep03 1e-9 200 2000 0.7"
"InfluenzaB 260605 ExponentialODipLinear kireldedupsac maxagep03 1e-9 200 2000 0.7"

"Metapneumovirus 260605 ExponentialODipLinear kireldedupsac maxagep02 1e-9 200 2000 0.7"
"Adenovirus 260605 ExponentialODipLinear kireldedupsac maxagep004 1e-9 200 2000 0.7"
"Parainfluenza3 260605 ExponentialODipLinear kireldedupsac maxagep005 1e-9 200 2000 0.7"
"InfluenzaA 260605 ExponentialODipLinear kireldedupsac maxagep04 1e-9 200 2000 0.7"
"InfluenzaB 260605 ExponentialODipLinear kireldedupsac maxagep04 1e-9 200 2000 0.7"

"Metapneumovirus 260605 ExponentialODipLinear unimmlimdedupsac maxagep01 1e-9 200 2000 0.7"
"Adenovirus 260605 ExponentialODipLinear unimmlimdedupsac maxagep002 1e-9 200 2000 0.7"
"Parainfluenza3 260605 ExponentialODipLinear unimmlimdedupsac maxagep003 1e-9 200 2000 0.7"
"InfluenzaA 260605 ExponentialODipLinear unimmlimdedupsac maxagep03 1e-9 200 2000 0.7"
"InfluenzaB 260605 ExponentialODipLinear unimmlimdedupsac maxagep03 1e-9 200 2000 0.7"

"Metapneumovirus 260605 ExponentialODipLinear unimmlimdedupsac maxagep02 1e-9 200 2000 0.7"
"Adenovirus 260605 ExponentialODipLinear unimmlimdedupsac maxagep004 1e-9 200 2000 0.7"
"Parainfluenza3 260605 ExponentialODipLinear unimmlimdedupsac maxagep005 1e-9 200 2000 0.7"
"InfluenzaA 260605 ExponentialODipLinear unimmlimdedupsac maxagep04 1e-9 200 2000 0.7"
"InfluenzaB 260605 ExponentialODipLinear unimmlimdedupsac maxagep04 1e-9 200 2000 0.7"
)

combination="${combinations[$SLURM_ARRAY_TASK_ID]}"

python -u Analyses/fit_opt.py $combination