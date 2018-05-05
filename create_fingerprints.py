
# coding: utf-8
import sys
from time import time
import numpy as np
import pandas as pd
import pickle
from scipy.sparse import csr_matrix
import utils.fingerprints as finger
from scipy import spatial



def create_word_fingerprint(_word):
    H, W, N, rows = 64, 64, 1000, 305554    # Network height, width and unit dimensions
    som_type = 'BSOM'
    sufix = '_'+som_type+'_'+str(H)+'_'+str(N)+'_'+str(rows)

    codebook = np.load('./serializations/codebook'+sufix+'.npy')
    loader = np.load('./serializations/X'+sufix+'.npz')
    X = csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])

    print (' loading snippets_by_word'+sufix)   
    with open('./serializations/snippets_by_word'+sufix+'.pkl', 'rb') as handle:
        snippets_by_word = pickle.load(handle)
    print ('./serializations/snippets_by_word'+sufix+' load done')

    H, W, N, rows = 64, 64, 1000, 305554    # Network height, width and unit dimensions
    som_type = 'BSOM'
    sufix = '_'+som_type+'_'+str(H)+'_'+str(N)+'_'+str(rows)

    a_original, a_sparse = finger.create_fingerprint(_word, snippets_by_word, codebook, X, H, W, sufix)




def main(_word):
    t1=time()
    create_word_fingerprint(_word)
    t2=time()
    print("\nTime taken for processing snippets\n----------------------------------------------\n{} s".format((t2-t1)))

if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        print ("wrong number of arguments")
        print ("python .\process_snippets.py <word>")
        sys.exit()
    main(sys.argv[1])


    




    



