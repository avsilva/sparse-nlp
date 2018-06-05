#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os
from PIL import Image
from PIL import ImageOps
import numpy as np
import operator
from functools import reduce
from sparse_som import *
import utils.corpora as corp
import utils.database as db
import concurrent.futures
import math

try:
    import conf.conn as cfg
except:
    print('ERROR: ' +str(sys.exc_info()[0]))



def sparsify_fingerprint(a):
    
    hist = np.histogram(a, bins=10, range=None, normed=False, weights=None, density=None)
    #print (hist[0])
    #print (hist[1])

    sparsify_percentage = 0.02
    nvalues = a.shape[0] * a.shape[1]
    maxpixels = nvalues * sparsify_percentage
    #print ('maxpixels:', maxpixels)

    actual_value = 0
    pixels_on = 0
    for idx, val in enumerate(reversed(hist[0])):
        actual_value += val
        if actual_value > maxpixels:
            lower_limit_index = idx
            break
        else:
            pixels_on += val

    #print ('pixels_on: ', pixels_on)
    pixels_on_missing = round(maxpixels - pixels_on)
    #print ('pixels_on_missing: ', pixels_on_missing) 
    #print ('lower_limit_index: ', lower_limit_index)

    rev = list(reversed(hist[1]))


    if pixels_on_missing > 0:
        print ('filling missing pixels...')
        #print ('lower count: ', rev[lower_limit_index + 1])
        #print ('higher count: ', rev[lower_limit_index + 0])
        lower = rev[lower_limit_index + 1]
        higher = rev[lower_limit_index + 0]
        a_copy = np.copy(a)
        counter = 0
        for x in np.nditer(a_copy, op_flags=['readwrite']):
            if counter < pixels_on_missing:
                if x >= lower and x < higher:
                    x[...] = 1
                    counter += 1
                    
                else:
                    x[...] = 0 
            else:
                x[...] = 0

    lower_count = rev[lower_limit_index]
    #print ('lower_count: ', lower_count)

    sparsify = lambda t: 1 if t > lower_count else 0
    vfunc = np.vectorize(sparsify)
    b = vfunc(a)

    total = np.sum([a_copy, b], axis=0)
    return total


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
    
    #im = Image.fromarray(a)
    im = Image.fromarray(a.astype('uint8'))

    if os.path.exists("./images/"+_word+_sufix+".bmp"):
        print ('Removing '+'./images/'+_word+_sufix+'.bmp')
        os.remove("./images/"+_word+_sufix+".bmp")
    
    im.save("./images/"+_word+_sufix+".png")
    Image.open("./images/"+_word+_sufix+".png").convert('RGB').save("./images/"+_word+_sufix+".bmp")
    
    
    im = Image.open("./images/"+_word+_sufix+".bmp")
    # calculate lookup table
    lut = equalize(im.histogram())
    # map image through lookup table
    im = im.point(lut)
    im.save("./images/"+_word+_sufix+".bmp")
    os.remove("./images/"+_word+_sufix+".png")


def create_fingerprint(_word, _snippets_by_word, som, X, sufix):
    
    
    W = som.ncols
    H = som.nrows
    #som = Som(H, W, N, topology.RECT, verbose=True) # , verbose=True
    #som.codebook = _codebook

    word_counts = _snippets_by_word[_word] 

    a = np.zeros((H, W), dtype=np.int)
    print ('######## Creating fingerprint: "' +str(_word)+ '" with '+str(len(word_counts))+' appearances in snippets  ########')
    i = 0
    for snippet_count in word_counts:
        #idx =  _dataframe.index[_dataframe['id'] == snippet_count['snippet']].tolist()[0]
        idx = snippet_count['idx']
        bmus = som.bmus(X[idx])
        #print (type(bmus))
        #print (bmus)
        #print (bmus[0].tolist())
        #bmus = [[93,  72]]
        
        #db.update_bmu (cfg, snippet_count['snippet'], int(bmus[0][0]), int(bmus[0][1]))

        a[bmus[0][0], bmus[0][1]] += snippet_count['counts']
        #if mode == 'binary':
            #a[bmus[0][0], bmus[0][1]] = 1
        #else:
            #a[bmus[0][0], bmus[0][1]] = snippet_count['counts']
        i +=1
        if i % 200 == 0:
            print ('########  snippet number: '+str(i)+'  ########')

    print ('########  fingerprint done - 1st phase  ########')
    
    sparse_fp = sparsify_fingerprint(a)
    create_fp_image (sparse_fp, _word, sufix)

    return [a, sparse_fp]
    
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