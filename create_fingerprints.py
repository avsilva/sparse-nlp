
# coding: utf-8
import sys
import os.path
from time import time
import numpy as np
import pandas as pd
import pickle
from scipy.sparse import csr_matrix
import utils.fingerprints as finger
import utils.corpora as corp
import utils.database as db
from scipy import spatial
from sparse_som import *

DIR = '/opt/sparse-nlp/datasets'
#DIR = 'C:/Users/andre.silva/web_data/'


def create_word_fingerprint(_word, _snippets_by_word, _codebook, X, H, W, _sufix):
    a_original, a_sparse = finger.create_fingerprint(_word, _snippets_by_word, _codebook, X, H, W, _sufix)


def  get_words_for_men_dataset(line):
    
    words = line.split(' ')
    w1 = words[0].split('-')[0]
    w2 = words[1].split('-')[0]
    #score = words[2].replace('\n', '')
    return [w1, w2]
    #return [w1, w2, score]

def fetch_MEN(_snippets_by_word, _codebook, X, H, W, _sufix):
    print ('fetching MEN dataset')
    filepath = DIR+'/similarity/EN-MEN-LEM.txt'
    if (os.path.isfile(filepath) == False):
        print ('FILE DOES NOT EXISTS')
        sys.exit(0)
    else:
        file = open(filepath, 'r', encoding='utf-8') 
        oov = 0
        oov_words = []
        for line in file:
            
            data = get_words_for_men_dataset(line)
            for w in data:
                fingerprintpath = './images/'+w+_sufix+'.bmp'
                if (os.path.isfile(fingerprintpath) == False):
                    print ("Linha: %s  "% (nline))
                    #create_word_fingerprint(w, _snippets_by_word, _codebook, X, H, W, _sufix)
                    a_original, a_sparse = finger.create_fingerprint(w, _snippets_by_word, _codebook, X, H, W, _sufix)
            w1 = data[0]
            w2 = data[1]
            
            if w1.lower() not in _snippets_by_word or w2.lower() not in _snippets_by_word:
                oov += 1
                oov_words.append(w1 + ' '+ w2)
            
        print ("There are %s OOV words in MEN dataset"% (oov))

def fetch_WS353(_snippets_by_word, _codebook, X, H, W, _sufix):
    print ('fetching WS353 dataset')
    filepath = DIR+'/similarity/EN-WS353.txt'
    if (os.path.isfile(filepath) == False):
        print ('FILE DOES NOT EXISTS')
        sys.exit(0)
    else:
        file = open(filepath, 'r', encoding='utf-8') 
        nline = 1
        oov = 0
        oov_words = []
        folder = 'BSOM_64_1000_305554'
        for line in file:
            
            if nline != 1:
                words = line.split('\t')[0:2]
                #score = words[2].replace('\n', '')
                
                for w in words:
                    fingerprintpath = './images/'+w+_sufix+'.bmp'
                    if (os.path.isfile(fingerprintpath) == False):
                        print ("Linha: %s %s "% (nline,  w.lower()))
                        #create_word_fingerprint(w.lower(), _snippets_by_word, _codebook, X, H, W, _sufix)
                        a_original, a_sparse = finger.create_fingerprint(w.lower(), _snippets_by_word, _codebook, X, H, W, _sufix)
                w1 = words[0]
                w2 = words[1]
                
                if w1.lower() not in _snippets_by_word or w2.lower() not in _snippets_by_word:
                    oov += 1
                    oov_words.append(w1 + ' '+ w2)
                
            nline += 1
            
        print ("There are %s OOV words in MEN dataset"% (oov))

def fetch_SimLex999():
    pass


def main(_word):
    
    # Define datasets
    datasets = {
        "men-dataset": fetch_MEN,
        "WS353-dataset": fetch_WS353,
        "SIMLEX999-dataset": fetch_SimLex999
    }

    # H, W, N, rows = 64, 64, 1000, 305554    # Network height, width and unit dimensions
    # som_type = 'BSOM'
    # sufix = '_'+som_type+'_'+str(H)+'_'+str(N)+'_'+str(rows)

    H, W, N, rows, algo = 64, 64, 2715, 591577, 'SDSOM'
    som_type = {'SDSOM': Som, 'BSOM': BSom}
    sufix = '_'+algo+'_'+str(H)+'_'+str(N)+'_'+str(rows)

    codebook = np.load('./serializations/codebook'+sufix+'.npy')
    som = som_type[algo](H, W, N, topology.RECT, verbose=True)
    som.codebook = codebook

    loader = np.load('./serializations/X'+sufix+'.npz')
    X = csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])

    print('loading snippets_by_word{}'.format(sufix))  
    dataframe = db.get_cleaned_data()
    print('dataframe shape {} '.format(dataframe.shape))
    words = ['sun', 'sunlight', 'grape', 'vine', 'leaf', 'nature', 'colour', 'rainbow', 'morning', 'sunshine', 
            'bloom', 'daffodil', 'holiday', 'travel', 'snow', 'weather', 'banana', 'cherry', 'potato', 'salad', 
            'eat', 'strawberry']
    snippets_by_word = corp.get_snippets_and_counts(dataframe, words)

    t1 = time()
    if _word in datasets:
        print("creating fingerprints for all dataset "+_word)
        datasets[_word](snippets_by_word, codebook, X, H, W, sufix)
    else:
        print("creating fingerprint for word "+_word)
        # create_word_fingerprint(_word, snippets_by_word, codebook, X, H, W, sufix)
        # a_original, a_sparse = finger.create_fingerprint(_word, snippets_by_word, codebook, X, H, W, sufix)
        a_original, a_sparse = finger.create_fingerprint(_word, snippets_by_word, som, X, sufix)

    t2 = time()
    print("\nTime taken for processing snippets\n----------------------------------------------\n{} s".format((t2-t1)))

if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        print("wrong number of arguments")
        print("python .\process_snippets.py <word or dataset>")
        sys.exit()
    main(sys.argv[1])

# python create_fingerprints.py sunlight
# python create_fingerprints.py men-dataset
# python create_fingerprints.py WS353-dataset
    




    



