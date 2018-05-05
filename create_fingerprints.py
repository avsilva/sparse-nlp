
# coding: utf-8
import sys
import os.path
from time import time
import numpy as np
import pandas as pd
import pickle
from scipy.sparse import csr_matrix
import utils.fingerprints as finger
from scipy import spatial

DIR = '/opt/sparse-nlp/datasets'



def create_word_fingerprint(_word, _snippets_by_word, _codebook, X, H, W, _sufix):
    a_original, a_sparse = finger.create_fingerprint(_word, _snippets_by_word, _codebook, X, H, W, _sufix)



def  get_words_for_men_dataset(line):
    
    words = line.split(' ')
    w1 = words[0].split('-')[0]
    w2 = words[1].split('-')[0]
    score = words[2].replace('\n', '')
    return [w1, w2, score]

def fetch_MEN(_snippets_by_word, _codebook, X, H, W, _sufix):
    print ('fetching MEN dataset')
    filepath = DIR+'/similarity/EN-MEN-LEM.txt'
    if (os.path.isfile(filepath) == False):
        print ('FILE DOES NOT EXISTS')
        sys.exit(0)
    else:
        file = open(filepath, 'r', encoding='utf-8') 
        nline = 1
        oov = 0
        oov_words = []
        for line in file:
            
            if nline != 1:
                data = get_words_for_men_dataset(line)
                w1 = data[0]
                w2 = data[1]
                print ("Words %s %s "% (w1, w2))
                #score = data[2]
                create_word_fingerprint(w1, _snippets_by_word, _codebook, X, H, W, _sufix)
                create_word_fingerprint(w2, _snippets_by_word, _codebook, X, H, W, _sufix)

                if w1.lower() not in _snippets_by_word or w2.lower() not in _snippets_by_word:
                    oov += 1
                    oov_words.append(w1 + ' '+ w2)
            nline += 1
            
        print ("There are %s OOV words in MEN dataset"% (oov))

def fetch_WS353():
    pass

def fetch_SimLex999():
    pass


def main(_word):
    

    # Define datasets
    datasets = {
        "men-dataset": fetch_MEN,
        "WS353-dataset": fetch_WS353,
        "SIMLEX999-dataset": fetch_SimLex999
    }

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

    t1=time()
    if _word in datasets:
        print ("creating fingerprints for all dataset "+_word)
        datasets[_word](snippets_by_word, codebook, X, H, W, sufix)
    else:
        print ("creating fingerprint for word "+_word)
        create_word_fingerprint(_word, snippets_by_word, codebook, X, H, W, sufix)

    t2=time()
    print("\nTime taken for processing snippets\n----------------------------------------------\n{} s".format((t2-t1)))

if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        print ("wrong number of arguments")
        print ("python .\process_snippets.py <word or dataset>")
        sys.exit()
    main(sys.argv[1])

#python create_fingerprints.py ceramic
#python create_fingerprints.py men-dataset
    




    



