#!/usr/bin/env python3
import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import h5py
from glob import glob
import os
from random import sample
from tqdm import tqdm
import re
# fix random seed
np.random.seed(0)

from constants import *

# functions

# are runs valid
def is_run_valid(run):
    assert os.path.isdir(PREFIX_PROC + "r" + get_run_string(run))

# get files from run
def get_proc_files(run, tile):
    assert (tile > -1 and tile < 16)
    files = glob(PREFIX_PROC + "r" + get_run_string(run) + "/" + "CORR-R" + get_run_string(run) + "-AGIPD" + get_tile_string(tile) + "-S*.h5")
    return files
    
# photon number per file
def get_photon_number_per_file(file):
    agipd_tile = get_agipd_tile_from_file_name(file)
    f = h5py.File(file, 'r')
    if "/INSTRUMENT/MID_DET_AGIPD1M-1/DET/" + str(agipd_tile) + "CH0:xtdf/image/data" not in f:
        return None
    data = f["/INSTRUMENT/MID_DET_AGIPD1M-1/DET/" + str(agipd_tile) + "CH0:xtdf/image/data"]
    number_of_images = data.shape[0]
    samples = sample(range(number_of_images), SAMPLES_PER_TILE)
    photon_numbers = np.zeros(len(samples), dtype=np.int32)
    i = 0
    for sampleo in samples:
        image_data = np.array(data[sampleo]).ravel()
        photon_numbers[i] = get_photon_number_per_image(image_data)
        i += 1
    return rms(photon_numbers)
        
def get_photon_number_per_image(image_data):
    photons = np.histogram(image_data, bins=np.arange(ENERGY_LOW, ENERGY_HIGH, ENERGY_STEP))[0]
    return np.sum(photons)
    
# aux functions
def get_run_string(run):
    return str(run).zfill(4)

def get_tile_string(tile):
    assert (tile > -1 and tile < 16)
    return str(tile).zfill(2)

def rms(data):
    return np.sqrt(np.mean(data**2))

def get_all_files_from_runs(runs):
    all_files_from_run = []
    for run in runs:
        is_run_valid(run)
        for i in TILES:
            for element in get_proc_files(run, i):
                all_files_from_run.append(element)
    return all_files_from_run

def get_agipd_tile_from_file_name(filename):
    try:
        found = re.search('AGIPD(.+?)-S00', filename).group(1)
        return int(found)
    except AttributeError:
        print("Could not extract AGIPD tile number from file name")

def main():
    parser = argparse.ArgumentParser(description='Probabilistic Evaluator of detector hit rate')
    parser.add_argument('run', help='Run number', type=int)
    args = parser.parse_args()

    run = int(args.run)

    all_files = get_all_files_from_runs([run])
    photon_numbers = []
    for i in tqdm(range(len(all_files))):
        number_per_file = get_photon_number_per_file(all_files[i])
        if (number_per_file is None):
            continue
        photon_numbers.append(number_per_file)

    mean = np.mean(photon_numbers)*len(TILES)
    print("mean photon number per image for run: " + str(run) + " is " + str(mean))
   
if __name__ == '__main__':
    main() 
    

