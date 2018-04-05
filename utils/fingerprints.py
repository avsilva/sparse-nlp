#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from PIL import Image
import numpy as np
from sparse_som import *
import utils.corpora as corp

def create_fingerprint(_word, _dataframe, _codebook, X, H, W, mode):
    
    try:
        sufix = ''
        N = X.shape[1]
        som = Som(H, W, N, topology.RECT, verbose=True) # , verbose=True
        som.codebook = _codebook

        word_counts_per_snippet = corp.get_word_counts_per_snippet(_dataframe, clean_text=False)
        snippets_by_word = corp.get_snippets_by_word(word_counts_per_snippet)

        word_counts = snippets_by_word[_word]

        a = np.zeros((H, W), dtype=np.int)
        print ('########  ' +str(_word)+ '  ########')
        for snippet_count in word_counts:
            idx =  _dataframe.index[_dataframe['id'] == snippet_count['snippet']].tolist()[0]
            bmus = som.bmus(X[idx])
            #print (bmus[0].tolist())
            if mode == 'binary':
                sufix = '_'
                a[bmus[0][0], bmus[0][1]] = 1
            else:
                a[bmus[0][0], bmus[0][1]] = snippet_count['counts']
        im = Image.fromarray(a)
        im.save("./images/"+_word+sufix+".png")
    except :
        print('ERROR ' + str(sys.exc_info()[0]))