import os, json, sys, csv, shutil
from time import time
import pandas as pd
import ast
from sqlalchemy import create_engine
from PIL import Image
import pickle
import datetime
import utils.database as db
import conf.conn as cfg
import utils.corpora as corp
import utils.fingerprints as finger
import utils.evaluation as eval

import operator
from scipy.sparse import csr_matrix
from sparse_som import *
from minisom import MiniSom
import numpy as np
import re
import concurrent.futures
import math


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import safe_indexing

# http://masnun.com/2016/03/29/python-a-quick-introduction-to-the-concurrent-futures-module.html

#DIR = './benchmarck/similarity'
DIR = 'C:/Users/andre.silva/web_data/'

"""
DATASET = 'RGB65'
H, W, N, rows, algo = 64, 64, 700, 15987, 'SDSOM'
som_type = {'SDSOM': Som, 'BSOM': BSom}
sufix = '_'+algo+'_'+str(H)+'_'+str(N)+'_'+str(rows)
LIMIT = 1000

codebook = np.load('./serializations/codebook'+sufix+'.npy')
SOM = som_type[algo](H, W, N, topology.RECT, verbose=True)
SOM.codebook = codebook

loader = np.load('./serializations/X'+sufix+'.npz')
X = csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])
"""

def fetch_ENRG65():
    filepath = DIR+'/similarity/EN-RG-65.txt'
    file = open(filepath, 'r', encoding='utf-8')
    score = []
    w1 = []
    w2 = []
    for line in file:
        data = eval.get_words_for_rg65_dataset(line)
        w1.append(data[0])
        w2.append(data[1])
        score.append(data[2])

    words = w1 + w2
    dictionary = set(words)
    return dictionary


def get_winner(data):
    global X
    global SOM
    idx = data['idx']
    bmus = SOM.winner(X[idx])
    return bmus   


def get_bmu(data):
    global X
    global SOM
    idx = data['idx']
    bmus = SOM.bmus(X[idx])
    return bmus


def main(_args):
    
    global SOM
    global X
    
    dataset = _args[1]
    mode = _args[2]
    logid = _args[3]
    loginfo = db.get_loginfo(cfg, logid)
    opts = ast.literal_eval(loginfo[0][2])
    
    """
    algo = suffix.split('_')[0]
    H = int(suffix.split('_')[1])
    W = int(suffix.split('_')[1])
    N = int(suffix.split('_')[2])
    rows = int(suffix.split('_')[3])
    """
    
    algo = loginfo[0][1]
    H = int(opts['size'])
    W = int(opts['size'])
    N = int(opts['ndimensions'])
    rows = int(opts['nsnippets'])
    snippet_limit = int(opts['paragraph_length'])
    print(H, W, N, rows, algo)
    
    sufix = '_'+algo+'_'+str(H)+'_'+str(N)+'_'+str(rows)
    print('loading snippets_by_word_{}'.format(str(logid)+'.pkl')) 

    dictionary = fetch_ENRG65()
    words = list(dictionary)
    # print(words) 

    filepath = './serializations/snippets_by_word_'+str(logid)+'.pkl'
    if (os.path.isfile(filepath) == False):
        dataframe = db.get_cleaned_data(None, snippet_limit)
        print('dataframe shape {} '.format(dataframe.shape))
        snippets_by_word = corp.get_snippets_and_counts(dataframe, words)

        with open('./serializations/snippets_by_word_'+str(logid)+'.pkl', 'wb') as f:
            pickle.dump(snippets_by_word, f)
        
    #else:
    with open('./serializations/snippets_by_word_'+str(logid)+'.pkl', 'rb') as handle:
        snippets_by_word = pickle.load(handle)


    # do the actual fingerprint creation
    if mode != 'ALL':
        words = mode.split(',')
    
    for word in words:
        
        word_counts = snippets_by_word[word]
        print(word, len(word_counts))
        # print(word_counts)

        t1 = time()
        #result = []
        a = np.zeros((H, W), dtype=np.int)

        if algo in ['SDSOM', 'BSOM']: 

            som_type = {'SDSOM': Som, 'BSOM': BSom}
            SOM = som_type[algo](H, W, N, topology.RECT, verbose=True)
            codebook = np.load('./serializations/codebook_'+str(logid)+'.npy')
            SOM.codebook = codebook

            loader = np.load('./serializations/X_'+str(logid)+'.npz')
            X = csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])

            # with concurrent.futures.ProcessPoolExecutor() as executor:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:

                for info, bmu in zip(word_counts, executor.map(get_bmu, word_counts)):
                    print(info['counts'], bmu[0][0], bmu[0][1])
                    #result.append(bmu)
                    a[bmu[0][0], bmu[0][1]] += info['counts']

        elif algo in ['MINISOMBATCH', 'MINISOMRANDOM']:

            with open('./serializations/codebook_'+str(logid)+'.npy', 'rb') as handle:
                codebook = pickle.load(handle)
            SOM = MiniSom(H, W, N, sigma=1.0, random_seed=1)
            SOM.load_weights(codebook)
            #SOM._weights = codebook
            with open('./serializations/X_'+str(logid)+'.npz', 'rb') as handle:
                X = pickle.load(handle)

            for info in word_counts[1:]:
                # print (info)
                idx = info['idx']
                bmu = SOM.winner(X[idx])
                a[bmu[0], bmu[1]] += info['counts']
                #print (bmu, info)   
            """
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:

                for info, bmu in zip(word_counts, executor.map(get_winner, word_counts)):
                    print(info['counts'], bmu)
                    #result.append(bmu)
                    a[bmu[0], bmu[1]] += info['counts']
            """
        t2 = time()
        print("----------------------------------------------\nTime taken for processing snippets\n{} s".format((t2-t1)))

        np.savetxt('./tests/'+word+'.txt', a, fmt='%10.0f')
        #t1 = time()
        a = finger.sparsify_fingerprint(a)
        finger.create_fp_image(a, word, 'fp_'+str(logid))
        #t2 = time()
        #print("\nTime taken for creating image\n----------------------------------------------\n{} s".format((t2-t1)))
        

if __name__ == '__main__':
    

    if len(sys.argv) != 4:
        print ("wrong number of arguments")
        #print ("python .\create_fp_multiprocessing.py <dataset> <mode> <suffix> <len limit>")
        print ("python .\create_fp_multiprocessing.py <dataset> <mode> <id>")
        sys.exit()

    

    main(sys.argv)
    
# python create_fp_multiprocessing.py RGB65 ALL 3
# python create_fp_multiprocessing.py RGB65 car 6
# python create_fp_multiprocessing.py RGB65 automobile 6
# python create_fp_multiprocessing.py RGB65 wizard 6
