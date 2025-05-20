#!/bin/bash
#SBATCH --time=09:33:05
#SBATCH --partition=kipac,hns,normal
#SBATCH --ntasks=25 
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10000MB
source /home/users/kokron/Libraries/miniconda3/etc/profile.d/conda.sh

#conda init 
conda activate nbodykit-env
module load openmpi
cd /oak/stanford/orgs/kipac/users/mahlet/bias/scripts

mpirun -n 25 -w python helloworld.py 