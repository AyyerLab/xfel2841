#!/bin/bash


#SBATCH --array=146-167
#SBATCH --partition=upex-beamtime
#SBATCH --reservation=upex_002841
#SBATCH --time=01:00:00
#SBATCH --export=ALL
#SBATCH -J vds
#SBATCH -o .%j.out
#SBATCH -e .%j.out

# Change the runs to process using the --array option on line 3

PREFIX=/gpfs/exfel/exp/MID/202102/p002841/

source /etc/profile.d/modules.sh
source ${PREFIX}/scratch/ayyerkar/ana/source_this

run=`printf %.4d "${SLURM_ARRAY_TASK_ID}"`
extra-data-make-virtual-cxi ${PREFIX}/proc/r${run} -o ${PREFIX}/scratch/vds/r${run}_proc.cxi
