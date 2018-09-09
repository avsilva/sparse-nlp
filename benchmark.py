import datetime
import multiprocessing as mp
import pickle
import sys
import time
from math import sqrt
from timeit import default_timer as timer
import timeit
import dask.multiprocessing
#import findspark
import numba
import numba.cuda.api
import numba.cuda.cudadrv.libs
import numpy as np
from dask import compute, delayed
# https://github.com/numba/numba/tree/master/examples
# https://developer.nvidia.com/how-to-cuda-python
from minisom import MiniSom
from numba import cuda, jit, njit, vectorize
from numpy import (arange, array, dot, exp, linalg, logical_and, meshgrid,
                   nditer, outer, pi, power, random, subtract, unravel_index,
                   zeros)
from scipy import spatial

from sparsenlp.datacleaner import DataCleaner
from sparsenlp.fingerprint import FingerPrint
from sparsenlp.datasets import Datasets
from sparsenlp.sentencecluster import SentenceCluster
from sparsenlp.sentencesvect import SentenceVect

#numba.cuda.cudadrv.libs.test()
#numba.cuda.api.detect()

# nvprof python .\vectorAdd.py





def fast_norm_without_numba(x):
    """Returns norm-2 of a 1-D numpy array.
    * faster than linalg.norm in case of 1-D arrays (numpy 1.9.2rc1).
    """
    
    #idx = np.array(np.linalg.norm(x))
    #return idx
    return sqrt(np.dot(x, x.T))

@njit
def fast_norm(x):
    """Returns norm-2 of a 1-D numpy array.
    * faster than linalg.norm in case of 1-D arrays (numpy 1.9.2rc1).
    """
    
    #idx = np.array(np.linalg.norm(x))
    #return idx
    return sqrt(np.dot(x, x.T))

def _activate(codebook, x, with_numba):
    """Updates matrix activation_map, in this matrix
    the element i,j is the response of the neuron i,j to x"""
    _activation_map = zeros((128, 128))
    s = subtract(x, codebook)  # x - w
    it = nditer(_activation_map, flags=['multi_index'])

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
    return _activation_map

def find_nearest_vector(codebook, value, with_numba=True):
    
    
    _activation_map = _activate (codebook, value, with_numba)
    a = unravel_index(_activation_map.argmin(), _activation_map.shape)
    return a
    #a = codebook-value
    #return unravel_index(a.argmin(), a.shape)


def numba(codebook, word_vectors):
    
    #run_parallel = numba.config.NUMBA_NUM_THREADS > 1

    a = np.zeros((128, 128), dtype=np.int)

    for word in word_vectors:
        #print (word)
        for key, value in word.items():
            print (key, len(value))
            #print (key, type(value), len(value))
            for val in value:
                idx = val['idx']
                #bmu = SOM.winner(val['vector'])
                bmu2 = find_nearest_vector(codebook, val['vector'])
    #return {key: a}


var_dict = {}

def init_worker(H, W, N, codebook):
    # Using a dictionary is not strictly necessary. You can also
    # use global variables.
    var_dict['H'] = H
    var_dict['W'] = W
    var_dict['N'] = N
    var_dict['codebook'] = codebook

def create_fp(word_vectors):
        
    #SOM = MiniSom(var_dict['H'], var_dict['W'], var_dict['N'], sigma=1.0, random_seed=1)
    #SOM._weights = var_dict['codebook']
    a = np.zeros((var_dict['H'], var_dict['W']), dtype=np.int)

    for key, value in word_vectors.items():
        #print (key, len(value))
        for val in value:
            idx = val['idx']
            #bmu = SOM.winner(val['vector'])
            bmu2 = find_nearest_vector(var_dict['codebook'], val['vector'])
            #print(bmu)
            #a[bmu[0], bmu[1]] += val['counts']
            
    #return {key: a}

def multiprocess(codebook, word_vectors):

    num_processes = mp.cpu_count() - 1
    H = 128
    W = 128
    N = 50

    with mp.Pool(processes=num_processes, initializer=init_worker, initargs=(H, W, N, codebook)) as pool:
        results = pool.map(create_fp, word_vectors)


def process(codebook, word_vectors):
        
    #a = np.zeros((128, 128), dtype=np.int)

    for key, value in word_vectors.items():
        #print (key, len(value))
        for val in value:
            idx = val['idx']
            #bmu = SOM.winner(val['vector'])
            bmu2 = find_nearest_vector(codebook, val['vector'])

def calculate_cKDTree (codebook, x):
    
    distance, index = spatial.cKDTree(codebook).query(x)
    return index

def find_nearest_vector_ckdtree(codebook, x, H, W):
    
    #distance, index = spatial.cKDTree(codebook).query(x)
    index = calculate_cKDTree(codebook, x)
    bmu = unravel_index(index, codebook.shape)
    return bmu

def process_ckdtree(codebook, word_vectors, H, W):
    
    bmu = find_nearest_vector_ckdtree(codebook, word_vectors['vector'], H, W)
    
    #for key, value in word_vectors.items():
    #    for val in value:
    #        idx = val['idx']
    #        bmu2 = find_nearest_vector_ckdtree(codebook, val['vector'], H, W)


def dask(codebook, word_vectors):

    values = [delayed(process)(codebook, x) for x in word_vectors]
    #import dask.threaded
    #results = compute(*values, scheduler='threads')

    
    results = compute(*values, scheduler='processes')


def ckdtree(codebook, word_vectors):
    
    H = codebook.shape[0]
    W = codebook.shape[1]
    #print (word_vectors)
    print (len(word_vectors))
    #print (type(word_vectors))

    codebook = np.reshape(codebook, (codebook.shape[0] * codebook.shape[1], codebook.shape[2]))
    for x in word_vectors:
        #print (x['vector'])
        bmu = find_nearest_vector_ckdtree(codebook, x['vector'], H, W)

    

    """
    codebook = np.reshape(codebook, (16384, 50))
    values = [delayed(process_ckdtree)(codebook, x, H, W) for x in word_vectors]
    import dask.multiprocessing
    results = compute(*values, scheduler='processes')
    """

def sequential(codebook, word_vectors):

    a = np.zeros((128, 128), dtype=np.int)

    for word in word_vectors:
        for key, value in word.items():
            #print (key, len(value))
            for val in value:
                idx = val['idx']
                #bmu = SOM.winner(val['vector'])
                bmu2 = find_nearest_vector(codebook, val['vector'], False)

def main(mode, algo=None):

    #global codebook
    #SOM = MiniSom(128, 128, 50, sigma=1.0, random_seed=1)
    with open('/dev/shm/codebook_6.npy', 'rb') as handle:
        codebook = pickle.load(handle)
    with open('/dev/shm/X_6.npz', 'rb') as handle:
        X = pickle.load(handle)
    with open('/dev/shm/snippets_by_word_6_EN-RG-65.pkl', 'rb') as handle:
        snippets_by_word = pickle.load(handle)
        
    if mode == 'fingerprints':
        
        """
        opts = {'id': 26, 'paragraph_length': 300, 'dataextension': '3,4', 'n_features': 10000, 'n_components': 700, 
        'use_idf': False, 'use_hashing': False, 'use_glove': 'glove.6B.50d', 'algorithm': 'MINISOMBATCH', 
        'initialization': True, 'size': 128, 'niterations': 1000, 'minibatch': True, 'testdataset': 'EN-RG-65', 
        'verbose': False, 'date': '27-8-2018 9:51', 'create_vectors-minutes': 6.0, 'cluster-minutes': 7.0, 
        'create_fingerprints-minutes': 31.0, 'cosine': 0.632}
        """
        opts = {}
        opts['id'] = 6
        opts['new_log'] = False
        opts['sentecefolder'] = '/dev/shm/'
        opts['algorithm'] = 'MINISOMBATCH'
        #words = {'EN-RG-65': ['asylum', 'autograph', 'automobile']}
        words = {'EN-RG-65': ['asylum']}
        #vectors = SentenceVect(opts)
        #X = vectors.create_vectors()
        #snippets_by_word = vectors.create_word_snippets(words)
        #mycluster = SentenceCluster(opts)
        #codebook = mycluster.cluster(X)

        fingerprints = FingerPrint(opts, algo)
        fingerprints.create_fingerprints(snippets_by_word, X, codebook, fraction=1)
        
    else:

        #SOM._weights = codebook
        
        words = ['asylum']
        word_vectors = []
        unique_indexes = set()
        #print (words)

        for word in words:
            a = []
            word_counts = snippets_by_word[word]

            for info in word_counts[1:]:
                idx = info['idx']
                #print ('idx {}'.format(idx))
                a.append({'idx': idx, 'counts': info['counts'], 'vector': X[idx]})
                unique_indexes.add(idx)
            word_vectors.append({word: a})

        
        
        if mode == 'dask':


            print(codebook.shape)
            sys.exit(0)
            eval(mode)(codebook, word_vectors)

            

        elif mode == 'ckdtree':

            unique_word_vectors = []
            for idx in unique_indexes:
                unique_word_vectors.append({'idx': idx, 'vector': X[idx]})
        
            eval(mode)(codebook, unique_word_vectors) 


        #print (word_vectors[1]['asylum'][0]['vector'])
        
        
        
        

if __name__ == '__main__':

    time1 = datetime.datetime.now()
    mode = sys.argv[1]
    if mode == 'fingerprints':
        algo = sys.argv[2]
        main(mode, algo)
    else:
        main(mode)

    time2 = datetime.datetime.now()
    print(time2 - time1)
    
    
    


    """
    A = np.random.random((10, 10, 5))*100
    print (A)
    #pt = np.random.random((5))*100
    #pt = np.array([29.65562651, 99.20112434, 24.94200411, 10.59061549, 95.09526111])
    pt = np.array([49.03053465, 98.94097773,  6.53042072, 78.32344383, 28.83984973])
    print ('------------------')
    print(pt)
    B = np.reshape(A, (A.shape[0] * A.shape[1], A.shape[2]))
    print ('------------------')
    print (B.shape)
    print ('------------------')
    distance, index = spatial.cKDTree(B).query(pt)
    print(index)
    bmu = unravel_index(index, (A.shape[0], A.shape[1]))
    #bmu = unravel_index(index, B.shape)
    print(bmu)
    sys.exit(0)
    """

   


    
    # https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    # https://blog.krum.io/k-d-trees/
    # https://stackoverflow.com/questions/10818546/finding-index-of-nearest-point-in-numpy-arrays-of-x-and-y-coordinates
    

# python -W ignore .\benchmark.py ckdtree (12 seg)
# python -W ignore .\benchmark.py dask (29 seg)
# python -W ignore .\benchmark.py numba (38 seg)
# python -W ignore .\benchmark.py multiprocess (31 seg)
# python -W ignore .\benchmark.py sequential (70 seg)
# python -W ignore .\benchmark.py fingerprints numba
# python -W ignore .\benchmark.py fingerprints ckdtree

# nvprof python .\benchmark.py numba


# check https://towardsdatascience.com/trying-out-dask-dataframes-in-python-for-fast-data-analysis-in-parallel-aa960c18a915


