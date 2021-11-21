#!/bin/bash

#SBATCH --array=183
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --time=08:00:00
#SBATCH --export=ALL
#SBATCH -J proc_raw
#SBATCH -o .%j.out
#SBATCH -e .%j.out
#SBATCH --partition=upex-beamtime
#SBATCH --reservation=upex_002841

####SBATCH --partition=upex

# Change the runs to process using the --array option on line 3

source /etc/profile.d/modules.sh
module purge
module load anaconda3
source deactivate
source activate /gpfs/exfel/exp/MID/202102/p002841/usr/env

mask_file=/gpfs/exfel/exp/MID/202102/p002841/scratch/geom/mask_goodpix_cells_07.npy
cal_run=429
nevt=-1

mpirun -mca btl_tcp_if_include ib0 python ../proc_raw_modules.py $SLURM_ARRAY_TASK_ID -m ${mask_file} --calmd_run=${cal_run} -n $nevt
