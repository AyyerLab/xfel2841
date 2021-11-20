PREFIX = '/gpfs/exfel/exp/MID/202102/p002841/scratch/'
NCELLS = 352
CELL_SEL = slice(1,300,2)
ADU_PER_PHOTON = 8.
CHUNK_SIZE = 16

# configuration of countrate estimator
# define base path
PREFIX_PROC = "/gpfs/exfel/exp/MID/202102/p002841/scratch/proc/"

# specify runs
RUNS = [224]

# all energies in keV
ENERGY_LOW = 6.5
ENERGY_HIGH = 9.5
ENERGY_STEP = 0.25

# Blank out inner tiles, select tiles in general
TILES = [0,1,2,5,6,7,9,10,11,12,13,14]

# how many samples per tile?!
SAMPLES_PER_TILE = 10
