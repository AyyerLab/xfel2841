import sys
import os
import time
import argparse

import numpy as np
import h5py
from mpi4py import MPI

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
parser = argparse.ArgumentParser(description='Save photons to emc file')
parser.add_argument('run', help='Run number', type=int)
parser.add_argument('-n', '--nframes', help='Number of frames to process (default: -1, all)', type=int, default=-1)
parser.add_argument('-m', '--mask', help='Path to mask file (default: mask_goodpix_cells_02.npy)')
args = parser.parse_args()

# MPI stuff: Get which module to process
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()
assert nproc % 16 == 0
ranks_per_module = nproc // 16
my_module = rank // ranks_per_module
module_rank = rank % ranks_per_module

# Open VDS file
f_vds = h5py.File(PREFIX+'vds/r%.4d_proc.cxi' % args.run, 'r')
dset_vds = f_vds['entry_1/instrument_1/detector_1/data']
nevt = args.nframes if args.nframes >= 0 else dset_vds.shape[0]
num_chunks = nevt // CHUNK_SIZE
if rank == 0:
    print('Processing %d events with %d ranks per module in run %d' % (nevt, ranks_per_module, args.run))

# Create output datasets
f_out = h5py.File(PREFIX+'events/r%.4d_events.h5'%args.run, 'w', driver='mpio', comm=comm)
dset_p1 = f_out.create_dataset('entry_1/p1', (nevt, 16), dtype='f8')
dset_p2 = f_out.create_dataset('entry_1/p2', (nevt, 16), dtype='f8')
dset_imean = f_out.create_dataset('entry_1/imean', (nevt, 16), dtype='f8')

# Parse module geometry and masks
with h5py.File(PREFIX+'geom/geom_module.h5', 'r') as f:
    cx = f['x'][:].ravel()
    cy = f['y'][:].ravel()
if args.mask is None:
    mask = np.load(PREFIX+'geom/mask_goodpix_cells_02.npy')
else:
    mask = np.load(args.mask)
num_goodpix = mask.sum((2,3))

# Initialize g^(2) corr array
cshape = tuple(np.array((2*(cx.max() - cx.min()) + 1, 2*(cy.max()-cy.min()) + 1)).astype('i4'))
cen = np.array(cshape) // 2
corr = np.zeros((16,) + cshape)
powder = np.zeros((16, 512*128))
cell_mask = np.zeros(NCELLS, dtype='bool')
cell_mask[CELL_SEL] = True

my_num_chunks = int(np.ceil(num_chunks / ranks_per_module))
my_chunk_start = module_rank*my_num_chunks
my_chunk_end = np.clip((module_rank+1)*my_num_chunks, -1, num_chunks)
my_chunks = np.arange(my_chunk_start, my_chunk_end)

stime = time.time()
for i, ind in enumerate(my_chunks):
    s = ind * CHUNK_SIZE
    e = (ind+1) * CHUNK_SIZE
    chunk_cell_mask = cell_mask[np.arange(s, e) % NCELLS]
    chunk_ind = np.arange(s, e)[chunk_cell_mask]
    chunk_cid = chunk_ind % NCELLS
    frames = dset_vds[chunk_ind, my_module, :, :] * mask[chunk_cid, my_module]

    for j, frame_ind in enumerate(chunk_ind):
        if np.any(np.isnan(frames[j])):
            continue
        phot = np.clip(np.round(frames[j]/ADU_PER_PHOTON-0.3), 0, None).astype('i4').ravel()

        ngpix = num_goodpix[chunk_cid[j], my_module].sum()
        dset_p1[frame_ind, my_module] = (phot==1).sum() / ngpix
        dset_p2[frame_ind, my_module] = (phot==2).sum() / ngpix
        dset_imean[frame_ind, my_module] = phot.sum() / ngpix

        accumulate_corr(phot, corr, my_module)
        powder[my_module] += phot

    if rank == 0:
        sys.stderr.write('\rChunk %d/%d (%.3f Hz)' % (ind+1, my_num_chunks, (ind+1)*CHUNK_SIZE*ranks_per_module/(time.time()-stime)))
        sys.stderr.flush()
if rank == 0:
    sys.stderr.write('\nFinished processing in %.3f s\n' % (time.time()-stime))
    sys.stderr.flush()

f_out.close()
f_vds.close()
if rank == 0:
    sys.stderr.write('Closed events file in %.3f s\n' % (time.time()-stime))
    sys.stderr.flush()

if rank == 0:
    red_powder = np.zeros_like(powder)
    comm.Reduce(powder, red_powder, root=0, op=MPI.SUM)
    red_corr = np.zeros_like(corr)
    comm.Reduce(corr, red_corr, root=0, op=MPI.SUM)

    with h5py.File(PREFIX+'corr/r%.4d_corr.h5'%args.run, 'w') as f:
        f['data/corr'] = red_corr / nevt
        f['data/powder'] = red_powder.reshape(16,512,128) / nevt
        f['data/num_frames'] = nevt
else:
    comm.Reduce(powder, None, root=0, op=MPI.SUM)
    comm.Reduce(corr, None, root=0, op=MPI.SUM)

if rank == 0:
    sys.stderr.write('DONE in %.3f s\n' % (time.time()-stime))
    sys.stderr.flush()
