import sys
import os
import time
import argparse

import numpy as np
import h5py
from mpi4py import MPI
import yaml

from constants import *

def accumulate_corr(photons, corr, modnum):
    place = np.where(photons>0)[0]
    count = photons[place]

    x = cx[place]
    y = cy[place]
    diffx = np.rint(np.subtract.outer(x, x)).ravel().astype('i4') + cen[0]
    diffy = np.rint(np.subtract.outer(y, y)).ravel().astype('i4') + cen[1]
    count_prod = np.outer(count, count).ravel()
    np.add.at(corr[modnum], (diffx, diffy), count_prod)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Process raw data')
parser.add_argument('run', help='Run number', type=int)
parser.add_argument('-n', '--nframes', help='Number of frames to process (default: -1, all)', type=int, default=-1)
parser.add_argument('-m', '--mask', help='Path to mask file (default: mask_goodpix_cells_02.npy)')
parser.add_argument('--calmd_run', help='Run with calMD file (default: same as run)', type=int, default=-1)
args = parser.parse_args()

if args.calmd_run < 0:
    args.calmd_run = args.run

# MPI stuff: Get which module to process
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()
assert nproc % 16 == 0
ranks_per_module = nproc // 16
my_module = rank // ranks_per_module
module_rank = rank % ranks_per_module

# Open run
f_vds = h5py.File(PREFIX+'vds/r%.4d_raw.cxi' % args.run, 'r')
dset_vds = f_vds['entry_1/instrument_1/detector_1/data']
nevt = args.nframes if args.nframes >= 0 else dset_vds.shape[0]
if rank == 0:
    print('Processing %d events with %d ranks per module in run %d' % (nevt, ranks_per_module, args.run))

# Create output datasets
f_out = h5py.File(PREFIX+'events/raw/r%.4d_events.h5'%args.run, 'w', driver='mpio', comm=comm)
dset_p1 = f_out.create_dataset('entry_1/p1', (nevt, 16), dtype='f8')
dset_p2 = f_out.create_dataset('entry_1/p2', (nevt, 16), dtype='f8')
dset_imean = f_out.create_dataset('entry_1/imean', (nevt, 16), dtype='f8')

# Parse module geometry and masks
with h5py.File(PREFIX+'geom/geom_module.h5', 'r') as f:
    cx = f['x'][:].ravel()
    cy = f['y'][:].ravel()
if args.mask is None:
    mask_fname = PREFIX+'geom/mask_goodpix_cells_02.npy'
else:
    mask_fname = args.mask
mask = np.load(mask_fname)[:,my_module]
num_goodpix = mask.sum((1,2))

# Parse dark offsets
calmd_fname = PREFIX+'proc/r%.4d/calibration_metadata.yml' % args.calmd_run
with open(calmd_fname, 'r') as f:
    calmd = yaml.safe_load(f)
offset_fname = calmd['retrieved-constants']['AGIPD%.2d'%my_module]['constants']['Offset']['file-path']
with h5py.File(offset_fname, 'r') as f:
    offset = f[list(f.keys())[0]]['Offset/0/data'][:,:,:,0].transpose(2,1,0)

# Initialize g^(2) corr array
cshape = tuple(np.array((2*(cx.max() - cx.min()) + 1, 2*(cy.max()-cy.min()) + 1)).astype('i4'))
cen = np.array(cshape) // 2
corr = np.zeros((16,) + cshape)
powder = np.zeros((16, 512*128))
cell_mask = np.zeros(NCELLS, dtype='bool')
cell_mask[CELL_SEL] = True

my_nevt = int(np.ceil(nevt / ranks_per_module))
my_start = module_rank*my_nevt
my_end = np.clip((module_rank+1)*my_nevt, -1, nevt)
my_events = np.arange(my_start, my_end)

stime = time.time()
for i, ind in enumerate(my_events):
    cid = ind % NCELLS
    if not cell_mask[cid]:
        continue

    calib = dset_vds[ind, my_module, 0, :, :]
    if np.all(calib == 0):
        continue
    calib = calib.astype('f4') - offset[cid]
    calib = (calib.reshape(8,64,2,64) - np.median(calib.reshape(8,64,2,64), axis=(1,3), keepdims=True)).reshape(512,128)
    calib *= mask[cid]
    phot = np.clip(np.round(calib/RAW_ADU_PER_PHOTON-0.3), 0, None).astype('i4').ravel()

    dset_p1[ind, my_module] = (phot==1).sum() / num_goodpix[cid]
    dset_p2[ind, my_module] = (phot==2).sum() / num_goodpix[cid]
    dset_imean[ind, my_module] = phot.sum() / num_goodpix[cid]

    accumulate_corr(phot, corr, my_module)
    powder[my_module] += phot

    if rank == 0:
        sys.stderr.write('\rFrame %d/%d (%.3f Hz)' % (ind+1, my_nevt, (ind+1)*ranks_per_module/(time.time()-stime)))
        sys.stderr.flush()
if rank == 0:
    sys.stderr.write('\nFinished processing in %.3f s\n' % (time.time()-stime))
    sys.stderr.flush()

f_out.close()
if rank == 0:
    sys.stderr.write('Closed events file in %.3f s\n' % (time.time()-stime))
    sys.stderr.flush()

if rank == 0:
    red_powder = np.zeros_like(powder)
    comm.Reduce(powder, red_powder, root=0, op=MPI.SUM)
    red_corr = np.zeros_like(corr)
    comm.Reduce(corr, red_corr, root=0, op=MPI.SUM)

    with h5py.File(PREFIX+'corr/raw/r%.4d_corr.h5'%args.run, 'w') as f:
        f['data/corr'] = red_corr / nevt
        f['data/powder'] = red_powder.reshape(16,512,128) / nevt
        f['data/num_frames'] = nevt
        f['data/mask_fname'] = mask_fname
else:
    comm.Reduce(powder, None, root=0, op=MPI.SUM)
    comm.Reduce(corr, None, root=0, op=MPI.SUM)

if rank == 0:
    sys.stderr.write('DONE in %.3f s\n' % (time.time()-stime))
    sys.stderr.flush()
