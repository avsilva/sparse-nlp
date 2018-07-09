import os, json, sys, csv, shutil
from time import time
import pandas as pd
from sqlalchemy import create_engine
from PIL import Image
import pickle
import datetime
import utils.database as db
import utils.corpora as corp
import utils.fingerprints as finger

import operator
from scipy.sparse import csr_matrix
from sparse_som import *
from minisom import MiniSom
import numpy as np
import re
import conf.conn as cfg
import cluster

from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import safe_indexing




def load_codebook(sufix):
    sufix = '_'+sufix
    codebook = np.load('./serializations/codebook'+sufix+'.npy')
    return codebook

def get_train_texts(min_paragraph_length):
    print ("getting data from  dataframe ...")
    df_train = db.get_cleaned_data(None, min_paragraph_length)
    print ("get data done")
    train_data = df_train.cleaned_text
    return train_data

def load_vectors(sufix): 
    sufix = '_'+sufix
    loader = np.load('./serializations/X'+sufix+'.npz')
    X = csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])
    return X   

def generate_vectors(H, W, algo, min_paragraph_length):
    
    train_data = get_train_texts(min_paragraph_length)
    max_features=1000

    print ("vectorizing texts ...")
    vectorizer = TfidfVectorizer(
                                lowercase=True,
                                max_df=0.5, 
                                min_df=0.001, 
                                #max_features=N,  
                                stop_words='english', 
                                use_idf=True)
    X = vectorizer.fit_transform(train_data)
    print ("vectorizing texts done")

    N = X.shape[1]
    sufix = '_'+algo+'_'+str(H)+'_'+str(N)+'_'+str(X.shape[0])
    print (X.shape)
    print ("serializing sparse matrix X ...")
    np.savez('./serializations/X'+sufix+'.npz', data=X.data, indices=X.indices, indptr=X.indptr, shape=X.shape)


def generate_som(som_params, X, codebook):
    # setup SOM network
    som_type = {'SDSOM': Som, 'BSOM': BSom}

    algo = som_params['algo']
    H = int(som_params['H'])
    W = int(som_params['W'])
    N = int(X.shape[1])
    sufix = '_'+algo+'_'+str(H)+'_'+str(N)+'_'+str(X.shape[0])

    som = som_type[algo](H, W, N, topology.RECT, verbose=True) # , verbose=True
    # reinit the codebook (not needed) #try to use minisom to init the codebook based on data

    if codebook is not None:
        print ("loading initial codebook")
        som.codebook = codebook
    else:
        som.codebook = np.random.rand(H, W, N).astype(som.codebook.dtype, copy=False)

    print ("I'm training the SOM, so take a break, relax and have some coffee ...")
    time1 = datetime.datetime.now()
    som.train(X)
    time2 = datetime.datetime.now()
    elapsedTime = time2 - time1
    minutes = divmod(elapsedTime.total_seconds(), 60)[0]

    print ("serializing codebook ...")
    np.save('./serializations/codebook'+sufix+'_'+str(datetime.datetime.now().strftime("%d%m%y"))+'.npy', som.codebook)

    db.insert_log(cfg, algo+' trainning', sufix, minutes)

if __name__ == "__main__":
    
    if len(sys.argv) != 3:
        print ("wrong number of arguments")
        print ("python .\create_som.py <mode> <params>")
        sys.exit()
    
    mode = sys.argv[1]
    params = sys.argv[2]
    t1 = time()
    
    if mode == 'load_vectors':
        X = load_vectors(params)
        print (X.shape)
    elif mode == 'generate_vectors':
        generate_vectors(params)
    elif mode == 'generate_som_from_codebook':
        X = load_vectors(params)
        print (X.shape)
        codebook = load_codebook(params)
        algo = params.split('_')[0]
        som_size = params.split('_')[1]
        generate_som({'algo': algo, 'H': som_size, 'W': som_size}, X, codebook)
    elif mode == 'generate_som_from_scratch':
        algo = params.split('_')[0]
        som_size = params.split('_')[1]
        min_paragraph_length = params.split('_')[2]
        opts = {'n_components': 700,
            'use_hashing' : False,
            'use_idf' : True,
            'n_features' : 10000,
            'minibatch' : False,
            'verbose' : False}
        train_data = get_train_texts(min_paragraph_length)
        X = cluster.doc_representation(train_data, opts)
        
        N = X.shape[1]
        H = int(som_size)
        W = int(som_size)
        sufix = '_'+algo+'_'+str(H)+'_'+str(N)+'_'+str(X.shape[0])

        if algo in ['SDSOM', 'BSOM']:
            X = sparse.csr_matrix(X)
            np.savez('./serializations/X'+sufix+'.npz', data=X.data, indices=X.indices, indptr=X.indptr, shape=X.shape)
        
            som_type = {'SDSOM': Som, 'BSOM': BSom, 'MINISOM': MiniSom}
            som = som_type[algo](H, W, N, topology.RECT, verbose=True) # , verbose=True
            som.codebook = np.random.rand(H, W, N).astype(som.codebook.dtype, copy=False)
            som.train(X)
            np.save('./serializations/codebook'+sufix+'.npy', som.codebook)
        elif algo in ['MINISOMBATCH', 'MINISOMRANDOM']:
            
            niterations = int(params.split('_')[3])

            with open('./serializations/X'+sufix+'.npz', 'wb') as f:
                pickle.dump(X, f)
            
            N = X.shape[1]  # Nb. features (vectors dimension)
            print('number of features in SOM: {}'.format(N))
            som = MiniSom(H, W, N, sigma=1.0, random_seed=1)
            #som.random_weights_init(X)
           
            if algo == 'MINISOMBATCH':
                som.train_batch(X, niterations)
            elif algo == 'MINISOMRANDOM':
                som.train_random(X, niterations)
            
            with open('./serializations/codebook'+sufix+'.npy', 'wb') as f:
                pickle.dump(som.get_weights(), f)
        
    t2 = time()
    print("\nTime taken for processing snippets\n----------------------------------------------\n{} s".format((t2-t1)))

# python create_som.py load_vectors 'SDSOM_64_5545_571698'
# python create_som.py generate_som_from_codebook 'SDSOM_64_5545_571698'
# python create_som.py generate_som_from_scratch 'SDSOM_64_1000'

# python create_som.py generate_som_from_scratch 'MINISOMBATCH_64_750_1000'
# python create_som.py generate_som_from_scratch 'MINISOMBATCH_somsize_lengthlimit_niterations'