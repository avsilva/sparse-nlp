# coding: utf-8
import sys, os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.datasets.base import Bunch
import utils.fingerprints as finger
import utils.evaluation as eval

DIR = 'C:/Users/andre.silva/web_data/'

def get_fingerprint_from_image(_word, _fingerprints, binary=False):
    
    fp_arr = _fingerprints.split('_')
    basename = '_'.join([fp_arr[x] for x in range(4)])
    filepath = './images/'+_fingerprints+'/'+_word+'_'+basename+'.bmp'
    
    im = Image.open(filepath)
    r, g, b = im.split()
    pix = np.array(r)
    if binary:
        np.place(pix, pix>1, [1])
    #pix = pix.flatten()
    return pix

def fetch_ENRG65(_fingerprints):
    filepath = DIR+'/similarity/EN-RG-65.txt'
    file = open(filepath, 'r', encoding='utf-8')
    
    words = []
    
    for line in file:
        data = eval.get_words_for_rg65_dataset(line)
        words.append(data[0])
        words.append(data[1])

    
    words = set(words)

    fp_arr = _fingerprints.split('_')
    basename = '_'.join([fp_arr[x] for x in range(4)])
    sufix = '_'+basename
    
    for word in words:
        fp = get_fingerprint_from_image(word, _fingerprints, False)
        #try:
        sparse_fp = finger.sparsify_fingerprint(fp)
        finger.create_fp_image (sparse_fp, word, sufix)
        #except:
            #print (word)
        
  

    


if __name__ == "__main__":
    
    # Define datasets
    datasets = {
        "EN-RG-65": fetch_ENRG65
    }

    if len(sys.argv) != 3:
        print ("wrong number of arguments")
        print ("python .\sparsify.py <fingerprints> <dataset>")
        sys.exit()
    
    fingerprints = sys.argv[1]
    dataset = sys.argv[2]

    number_of_files = len([f for f in os.listdir('./images') if os.path.isfile('./images/'+f)])
    if number_of_files != 0:
        print ('images folder has files')
        sys.exit(0)

    datasets[dataset](fingerprints)

    

# python .\sparsify.py SDSOM_64_5545_571698_id23_size32 EN-RG-65