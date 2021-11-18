'''Module to ease exploration of data from VDS files
Only for interactive use (Don't build scripts using this!!)

Import in IPython or Jupyter
Use appropriate matplotlib magic command for plotting
'''

import sys
import os.path as op
import importlib
import itertools
import warnings

import numpy as np
from scipy.stats import binned_statistic 
import pylab as P
import h5py
import extra_geom

from constants import *

class Explorer():
    def __init__(self, run, geom_file='agipd_2746_v1.geom', raw=False):
        self._fvds = None
        self.open_run(run, raw)

        if op.exists(geom_file):
            self.parse_geom(geom_file)
        else:
            self.geom = None
            print('No geom file parsed!')

    def open_run(self, run, raw=False):
        self.run_num = run
        if self._fvds is not None:
            self._fvds.close()
        if raw:
            self._fvds = h5py.File(PREFIX+'vds/r%.4d_raw.cxi'%run, 'r')
        else:
            self._fvds = h5py.File(PREFIX+'vds/proc/r%.4d_proc.cxi'%run, 'r')
        self._dset = self._fvds['entry_1/instrument_1/detector_1/data']
        self.raw_data = raw
        print('VDS data set shape:', self._dset.shape)

    def parse_geom(self, geom_file):
        self.geom = extra_geom.AGIPD_1MGeometry.from_crystfel_geom(PREFIX + 'geom/' + geom_file)
        x, y, _ = self.geom.get_pixel_positions().transpose(3,0,1,2) / 236e-6
        self.rad = np.sqrt(x*x + y*y)

    def parse_dark(self, dark_run):
        '''Get dark offsets and bad pixel mask'''
        with h5py.File(PREFIX+'dark/r%.4d_dark.h5'%dark_run, 'r') as f:
            self.dark = f['data/mean'][:]
            self.dcells = f['data/cellId'][:]
            sigma = f['data/sigma'][:]
        self.mask = (sigma.mean(1) < 0.5) | (sigma.mean(1) > 1.5)

    @staticmethod
    def assemble_dense(frame, out=None):
        '''Frame assembly with no panel gaps
        Use imshow(out, origin='lower') to plot properly
        '''
        assert frame.shape == (16,) + MODULE_SHAPE
        if out is None:
            out = np.zeros((1024,1024), dtype=frame.dtype)
        out[512:,512:] = frame[:4][::-1,::-1,::-1].transpose(0,2,1).reshape(512,512)
        out[:512,512:] = frame[4:8][::-1,::-1,::-1].transpose(0,2,1).reshape(512,512)
        out[:512,:512] = frame[8:12][::-1].transpose(0,2,1).reshape(512,512)
        out[512:,:512] = frame[12:][::-1].transpose(0,2,1).reshape(512,512)
        return out

    def get_radavg(self, data):
        data = data.ravel()
        mask = (~self.mask.ravel()) & np.isfinite(data)
        return binned_statistic(self.rad.ravel()[mask], data[mask], bins=np.arange(self.rad.max()+1))

    @staticmethod
    def _iterating_median(v, tol=3):
        if len(v) == 0:
            return 0
        #vmin, vmax = v.min(), v.max()
        vmin, vmax = -2*tol, 2*tol
        vmed = np.median(v[(vmin < v) & (v < vmax)])
        vmed0 = vmed
        i = 0
        while True:
            vmin, vmax = vmed-tol, vmed+tol
            vmed = np.median(v[(vmin < v) & (v < vmax)])
            if vmed == vmed0:
                break
            else:
                vmed0 = vmed
            i += 1
            if i > 20:
                break
        return vmed

    def _common_mode(self, img, mask):
        """img should be substracted by the dark.
        img.shape == (X, Y), mask.shape == (X, Y)"""
        ig = img.astype('f8').copy()
        L = 64
        for i, j in itertools.product(range(ig.shape[0] // L),
                                      range(ig.shape[1] // L)):
            img = ig[i*64:(i+1)*64, j*64:(j+1)*64]
            m = mask[i*64:(i+1)*64, j*64:(j+1)*64]
            med = self._iterating_median(img[m].flatten())
            img -= med
        return ig

    def get_corr(self, i, cmod=False):
        if self.raw_data:
            if self.dark is None:
                raise AttributeError('Parse darks first to get corrected frame with raw data')
            cellid = self._fvds['entry_1/cellId'][i, 0]
            out = self._dset[i,:,0] - self.dark[:, np.where(cellid==self.dcells)[0][0]]
            if not cmod:
                return out
            return np.array([self._common_mode(out[i], ~self.mask[i]) for i in range(16)])
        return self._dset[i]

    def plot_frame(self, i, vmin=-20, vmax=200, cmod=False, **kwargs):
        frame = self.get_corr(i, cmod=cmod)
        if self.geom is None:
            print('No geometry. Plotting dense assembly')
            assem = self.assemble_dense(frame)
        else:
            assem = self.geom.position_modules_fast(frame)[0][:,::-1]
        P.imshow(assem, origin='lower', vmin=vmin, vmax=vmax, **kwargs)
        P.gca().set_facecolor('dimgray')
