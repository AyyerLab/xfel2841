#!/bin/bash

#SBATCH --array=55
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=64
#SBATCH --time=04:00:00
#SBATCH --export=ALL
#SBATCH -J proc_idi
#SBATCH -o .%j.out
#SBATCH -e .%j.out
#SBATCH --partition=upex

####SBATCH --partition=upex-beamtime
####SBATCH --reservation=upex_002841

# Change the runs to process using the --array option on line 3

source /etc/profile.d/modules.sh
module purge
module load anaconda3
source deactivate
source activate /gpfs/exfel/exp/MID/202102/p002841/usr/env

which mpirun
which python
which h5ls
#mpirun -mca btl_tcp_if_include ib0 python ../proc_modules.py $SLURM_ARRAY_TASK_ID -n 10000
mpirun -mca btl_tcp_if_include ib0 python ../proc_modules.py $SLURM_ARRAY_TASK_ID
