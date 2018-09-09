import numpy as np
from numba import njit
from numpy import (zeros, subtract, nditer, unravel_index)
from math import sqrt
from scipy import spatial

var_dict = {}


def init_worker(H, W, N, codebook):
    # Using a dictionary is not strictly necessary. You can also
    # use global variables.
    var_dict['H'] = H
    var_dict['W'] = W
    var_dict['N'] = N
    var_dict['codebook'] = codebook


def create_fp(word_vectors):
    SOM = MiniSom(var_dict['H'], var_dict['W'], var_dict['N'], sigma=1.0, random_seed=1)
    SOM._weights = var_dict['codebook']
    a = np.zeros((var_dict['H'], var_dict['W']), dtype=np.int)

    for key, value in word_vectors.items():
        #print (key, type(value), len(value))
        for val in value:
            idx = val['idx']
            bmu = SOM.winner(val['vector'])
            a[bmu[0], bmu[1]] += val['counts']
            
    return {key: a}

@njit
def fast_norm(x):
    """Returns norm-2 of a 1-D numpy array.
    * faster than linalg.norm in case of 1-D arrays (numpy 1.9.2rc1).
    """
    
    #idx = np.array(np.linalg.norm(x))
    return sqrt(np.dot(x, x.T))


def _activate(codebook, x, H, W):
    """Updates matrix activation_map, in this matrix
    the element i,j is the response of the neuron i,j to x"""
    _activation_map = zeros((H, W))
    s = subtract(x, codebook)  # x - w
    it = nditer(_activation_map, flags=['multi_index'])

    while not it.finished:
        # || x - w ||
        _activation_map[it.multi_index] = fast_norm(s[it.multi_index])
        it.iternext()
    """
    if with_numba is True:
        while not it.finished:
            # || x - w ||
            _activation_map[it.multi_index] = fast_norm(s[it.multi_index])
            it.iternext()
    else:
        while not it.finished:
            # || x - w ||
            _activation_map[it.multi_index] = fast_norm_without_numba(s[it.multi_index])
            it.iternext()
    """
    return _activation_map


def find_nearest_vector(codebook, value, H, W):
    _activation_map = _activate (codebook, value, H, W)
    a = unravel_index(_activation_map.argmin(), _activation_map.shape)
    return a


def process2(codebook, word_vectors, H, W):
    
    bmu = find_nearest_vector(codebook, word_vectors['vector'], H, W)
    return {word_vectors['idx']: bmu}

def process3(codebook, word_vectors, H, W):
    
    a = np.zeros((H, W), dtype=np.int)
    for key, value in word_vectors.items():
        for val in value:
            bmu = find_nearest_vector(codebook, val['vector'], H, W)
            a[bmu[0], bmu[1]] += val['counts']
    return {key: a}


def calculate_cKDTree (codebook, x):
    distance, index = spatial.cKDTree(codebook).query(x)
    return index

def find_nearest_vector_ckdtree(codebook, x, H, W):
    
    #distance, index = spatial.cKDTree(codebook).query(x)
    index = calculate_cKDTree(codebook, x)
    bmu = unravel_index(index, (H, W))  
    return bmu


def process_ckdtree(codebook, word_vectors, H, W):

    bmu = find_nearest_vector_ckdtree(codebook, word_vectors['vector'], H, W)
    return {word_vectors['idx']: bmu}




    