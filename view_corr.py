import argparse

import numpy as np
import h5py
from scipy import signal
import extra_geom

from constants import *

def assem_mod(mod):
    assert mod.shape == (512, 128)
    assem = np.zeros((526, 128), dtype=mod.dtype)
    for i in range(8):
        assem[i*66:i*66+64] = mod[i*64:(i+1)*64]
    return assem
    
def normalize(corr, powder):
    apowder = np.array([assem_mod(mod) for mod in powder])
    cq = np.array([signal.fftconvolve(mod, mod[::-1,::-1]) for mod in apowder])
    return corr / cq

def extract_mod(corr, center=False):
    if center:
        return corr[:,301:813,63:191]
    else:
        return corr[:,269:781,63:191]

def plot_corrdet(corrdet, geom, vmax=None):
    import pylab as P
    geom = extra_geom.AGIPD_1MGeometry.from_crystfel_geom(geom)
    assem = geom.position_modules_fast(corrdet)[0]

    P.figure(figsize=(10,12))

    vmax = None if vmax is None else np.abs(vmax)
    vmin = None if vmax is None else -0.1*np.abs(vmax)
    im = P.imshow(assem[:,::-1], vmax=vmax, vmin=vmin)

    P.gca().set_facecolor('dimgray')
    P.colorbar(im, fraction=0.046*assem.shape[0]/assem.shape[1], pad=0.04)
    P.tight_layout()
    P.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize corr file for run')
    parser.add_argument('run', help='Run number', type=int)
    #parser.add_argument('-a', '--assemble', help='Assemble corr by module', action='store_true')
    parser.add_argument('-s', '--subtract', help='Subtract g^(2) offset for each module', action='store_true')
    parser.add_argument('-c', '--center', help='Shift q=0 to center of ASIC', action='store_true')
    parser.add_argument('-g', '--geom', help='Path to geometry file (default: agipd_2746_v1.geom)',
                        default=PREFIX+'geom/agipd_2746_v1.geom')
    parser.add_argument('--vmax', help='Color map vmax (default: auto)', type=float)
    args = parser.parse_args()

    with h5py.File(PREFIX+'corr/r%.4d_corr.h5'%args.run, 'r') as f:
        corr = f['data/corr'][:]
        powderc = f['data/powder'][:]
        
    ncorr = normalize(corr, powderc)
    corrdet = extract_mod(ncorr, center=args.center)

    if args.subtract:
        corrdet -= np.median(corrdet, axis=(1,2), keepdims=True)

    plot_corrdet(corrdet, args.geom, args.vmax)

if __name__ == '__main__':
    main()
