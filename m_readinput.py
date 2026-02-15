import numpy as np

def read_pos(filename):
    pos = np.loadtxt(filename)
    natoms = pos.shape[0]
    
    return pos, natoms 