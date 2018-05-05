#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.datasets.base import Bunch


def sparsify_fingerprint(a):
    
    hist = np.histogram(a, bins=10, range=None, normed=False, weights=None, density=None)
    print (hist)

    sparsify_percentage = 0.05
    nvalues = a.shape[0] * a.shape[1]
    limit = nvalues * sparsify_percentage

    actual_value = 0
    for idx, val in enumerate(reversed(hist[0])):
        #print(idx, val)   
        actual_value += val
        if actual_value > limit:
            limit_index = idx
            break
            

    if limit_index != 0:
        limit_index = limit_index - 1       
    #print (limit_index)

    rev = list(reversed(hist[1]))
    limit = rev[limit_index]

    #print ('limit:', limit)

    sparsify = lambda t: t if t > limit else 0
    binary = lambda t: 1 if t >= 1 else 0

    vfunc = np.vectorize(sparsify)
    b = vfunc(a)
    vfunc = np.vectorize(binary)
    c = vfunc(b)
    return c


def equalize(h):
    
    lut = []

    for b in range(0, len(h), 256):

        # step size
        step = reduce(operator.add, h[b:b+256]) / 255

        # create equalization lookup table
        n = 0
        for i in range(256):
            lut.append(n / step)
            n = n + h[i+b]

    return lut

def create_fp_image (a, _word, _sufix):
    im = Image.fromarray(a)
    im.save("./images/"+_word+_sufix+".png")
    
    Image.open("./images/"+_word+_sufix+".png").convert('RGB').save("./images/"+_word+_sufix+".bmp")
    
    
    im = Image.open("./images/"+_word+_sufix+".bmp")
    # calculate lookup table
    lut = equalize(im.histogram())
    # map image through lookup table
    im = im.point(lut)
    im.save("./images/"+_word+_sufix+".bmp")
    os.remove("./images/"+_word+_sufix+".png")

def create_fingerprint(_word, _dataframe, _snippets_by_word, _codebook, X, H, W, sufix):
    
    #try:
    
    N = X.shape[1]
    rows = X.shape[0]
    som = Som(H, W, N, topology.RECT, verbose=True) # , verbose=True
    som.codebook = _codebook

    #sufix = '_'+str(H)+'_'+str(N)+'_'+str(rows)

    #word_counts_per_snippet = corp.get_word_counts_per_snippet(_dataframe)
    #snippets_by_word = corp.get_snippets_by_word(word_counts_per_snippet)

    word_counts = _snippets_by_word[_word]

    a = np.zeros((H, W), dtype=np.int)
    print ('########  ' +str(_word)+ '  ########')
    for snippet_count in word_counts:
        idx =  _dataframe.index[_dataframe['id'] == snippet_count['snippet']].tolist()[0]
        bmus = som.bmus(X[idx])
        #print (bmus[0].tolist())

        a[bmus[0][0], bmus[0][1]] = snippet_count['counts']
        #if mode == 'binary':
            #a[bmus[0][0], bmus[0][1]] = 1
        #else:
            #a[bmus[0][0], bmus[0][1]] = snippet_count['counts']
    
    
    sparse_fp = sparsify_fingerprint(a)
    create_fp_image (sparse_fp, _word, sufix)
    
    """
    im = Image.fromarray(a)
    im.save("./images/"+_word+sufix+".png")
    
    Image.open("./images/"+_word+sufix+".png").convert('RGB').save("./images/"+_word+sufix+".bmp")
    
    
    im = Image.open("./images/"+_word+sufix+".bmp")
    # calculate lookup table
    lut = equalize(im.histogram())
    # map image through lookup table
    im = im.point(lut)
    im.save("./images/"+_word+sufix+".bmp")
    os.remove("./images/"+_word+sufix+".png")
    """

    #except :
        #print('ERROR ' + str(sys.exc_info()[0]))