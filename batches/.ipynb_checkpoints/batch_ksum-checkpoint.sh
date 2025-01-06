#!/bin/bash
#SBATCH --time=09:33:05
#SBATCH --partition=kipac,hns,normal
#SBATCH --ntasks=25 
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=6400MB

#conda init 
conda activate nbodykit-env
cd /oak/stanford/orgs/kipac/users/mahlet/bias/scripts

kmax=$1
nbias=$2
res=$3

for color in 'red' 'blue' 'total'
do
    for hodtype in 'TNG' 'UM'
    do
        for density in 'low' 'high' 'medium'
        do
            for z in '99' '67' '50' '40'
            do
                srun -n 25 -w $SLURM_JOB_NODELIST python mpi_ksum.py $hodtype $z $kmax $nbias $res $color $density
            done
        done
    done
done