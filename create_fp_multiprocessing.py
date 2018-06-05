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
import utils.evaluation as eval

import operator
from scipy.sparse import csr_matrix
from sparse_som import *
import numpy as np
import re
import concurrent.futures
import math


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import safe_indexing

# http://masnun.com/2016/03/29/python-a-quick-introduction-to-the-concurrent-futures-module.html

"""
print (' loading snippets_by_word'+sufix)   
dataframe = get_cleaned_data()
print ('dataframe ', dataframe.shape)
words = ['sun', 'sunlight', 'grape', 'vine', 'leaf', 'nature', 'colour', 'rainbow', 'morning', 'sunshine', 
         'bloom', 'daffodil', 'holiday', 'travel', 'snow', 'weather', 'banana', 'cherry', 'potato', 'salad', 
         'eat', 'strawberry']
snippets_by_word = corp.get_snippets_and_counts(dataframe, words)
print ('snippets_by_word: '+sufix+' load done')
"""


DIR = 'C:/Users/andre.silva/web_data/'
DATASET = 'RGB65'
H, W, N, rows, algo = 64, 64, 2715, 591577, 'SDSOM'
som_type = {'SDSOM': Som, 'BSOM': BSom}
sufix = '_'+algo+'_'+str(H)+'_'+str(N)+'_'+str(rows)

codebook = np.load('./serializations/codebook'+sufix+'.npy')
SOM = som_type[algo](H, W, N, topology.RECT, verbose=True)
SOM.codebook = codebook

loader = np.load('./serializations/X'+sufix+'.npz')
X = csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])


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
    

def clean(data):
    
    global X
    global SOM
    idx = data['idx']
    bmus = SOM.bmus(X[idx])
    return bmus


def main():
    
    #H, W, N, rows = 128, 128, 1000, 591577
    #som_type = 'BSOM'
    #sufix = '_'+som_type+'_'+str(H)+'_'+str(N)+'_'+str(rows)
    
    

    print('loading snippets_by_word{}'.format(sufix)) 

    dictionary = fetch_ENRG65()
    words = list(dictionary)

    filepath = './serializations/snippets_by_word_'+DATASET+'_'+str(rows)+'.pkl'
    if (os.path.isfile(filepath) == False):
        dataframe = db.get_cleaned_data()
        print('dataframe shape {} '.format(dataframe.shape))
        #words = ['sun', 'sunlight', 'grape', 'vine', 'leaf', 'nature', 'colour', 'rainbow', 'morning', 'sunshine', 
        #        'bloom', 'daffodil', 'holiday', 'travel', 'snow', 'weather', 'banana', 'cherry', 'potato', 'salad', 
        #        'eat', 'strawberry']

        
        snippets_by_word = corp.get_snippets_and_counts(dataframe, words)

        with open('./serializations/snippets_by_word_'+DATASET+'_'+str(rows)+'.pkl', 'wb') as f:
            pickle.dump(snippets_by_word, f)
    else:
        with open('./serializations/snippets_by_word_'+DATASET+'_'+str(rows)+'.pkl', 'rb') as handle:
            snippets_by_word = pickle.load(handle)

    

    for word in words:
        # word = 'leaf'
        
        word_counts = snippets_by_word[word]
        print(word, len(word_counts))

        t1 = time()
        result = []
        a = np.zeros((H, W), dtype=np.int)
        # with concurrent.futures.ProcessPoolExecutor() as executor:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for info, bmu in zip(word_counts, executor.map(clean, word_counts)):
                #print(info['counts'])
                result.append(bmu)
                a[bmu[0][0], bmu[0][1]] += info['counts']
        t2 = time()

        
        sparse_fp = finger.sparsify_fingerprint(a)
        finger.create_fp_image (sparse_fp, word, sufix)
        print("\nTime taken for processing snippets\n----------------------------------------------\n{} s".format((t2-t1)))

if __name__ == '__main__':
    main()
    