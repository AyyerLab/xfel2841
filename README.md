# p2841 - Investigation of the Coherence of X-ray Fluorescence

Beamtime analysis code repository

## Directory Structure
 * The root directory contains the main pipeline programs used to process the data. In addition it contains the environment source script.
 * The `slurm` directory contains SLURM sbatch scripts to run the pipeline on the Maxwell cluster
 * The `scripts` directory contains specific _ad hoc_ scripts used for post-processing

## Pipeline
The pipeline consists of two main programs:
 * VDS creation using `extra-data-make-virtual-cxi` from the EXtra-data package, creating `vds/r***_proc.cxi`
 * Run processing using `proc_modules.py` which creates two files
 	* `corr/r****_corr.h5`: Contains module-wise frame integrals and frame autocorrelation integrals
	* `events/r****_events.h5`: Values of `P_1`, `P_2` and `Imean` for each frame and module


