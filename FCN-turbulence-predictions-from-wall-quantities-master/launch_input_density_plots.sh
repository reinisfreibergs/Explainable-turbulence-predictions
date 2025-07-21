#!/usr/bin/env bash
#SBATCH --job-name=reinis_main_v3
#SBATCH -A NAISS2025-5-144 -p alvis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=T4:1
#SBATCH --time=01:00:00
#SBATCH --output=./logs/job_%j.out
#SBATCH --error=./logs/job_%j.error

set -x

cd /mimer/NOBACKUP/groups/deepmechalvis/reinis

module purge
module load TensorFlow/2.7.1-foss-2021b-CUDA-11.4.1
module load scikit-learn/1.0.1-foss-2021b
module load numba/0.54.1-foss-2021b-CUDA-11.4.1
module load matplotlib/3.5.2-foss-2021b

source /mimer/NOBACKUP/groups/deepmechalvis/reinis/reinis_shap_env/bin/activate

srun python main_v3.py
srun python input_density_plots_cleaned.py --yp 15 --ret 550
srun python input_density_plots_cleaned.py --yp 50 --ret 550