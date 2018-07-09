# coding: utf-8
import sys
import os.path
from time import time
import numpy as np
import pandas as pd
import pickle
import scipy
from scipy.sparse import csr_matrix
import utils.fingerprints as finger
import utils.evaluation as eval
from scipy.spatial import distance
from sklearn.datasets.base import Bunch
from PIL import Image
import utils.database as db
import conf.conn as cfg

#from skimage.measure import structural_similarity as ssim
from skimage.measure import compare_ssim  as ssim
#import cv2


# DIR = '/opt/sparse-nlp/datasets'
DIR = 'C:/Users/andre.silva/web_data/'


# def create_word_fingerprint(_word, _snippets_by_word, _codebook, X, H, W, _sufix):
# a_original, a_sparse = finger.create_fingerprint(_word, _snippets_by_word, _codebook, X, H, W, _sufix)

def cosine(A, B):
    # return np.array([v1.dot(v2.T)/(np.linalg.norm(v1)*np.linalg.norm(v2)) for v1, v2 in zip(A, B)])
    return np.array([1 - distance.cosine(v1, v2) for v1, v2 in zip(A, B)])


# spatial.distance.cosine computes the distance, and not the similarity. 
# must subtract the value from 1 to get the similarity.
def euclidean(A, B):
    return np.array([1 / distance.euclidean(v1, v2) for v1, v2 in zip(A, B)])


def struc_sim(A, B):
    return np.array([ssim(v1, v2) for v1, v2 in zip(A, B)])
    
    

def similarbits(A, B):
    C = A*B
    return np.array([ v1.sum() for v1 in C])  # sum number of 1 bits
    

def get_fingerprint_from_image(_word, _fingerprints, _mode):
    
    # fp_id = _fingerprints
    fp_arr = _fingerprints.split('_')
    basename = '_'.join([fp_arr[x] for x in range(4)])
    filepath = './images/'+_fingerprints+'/'+_word+'_'+basename+'.bmp'
    
    im = Image.open(filepath)
    r, g, b = im.split()
    pix = np.array(r)

    if _mode == '2darray':
        result = pix
    elif _mode == 'flatten':
        np.place(pix, pix>1, [1])
        pix = pix.flatten()
        result = pix

    return result


def fetch_MEN(_fingerprints, _distance_measure, _percentage):
    print('fetching MEN dataset')
    filepath = DIR+'/similarity/EN-MEN-LEM.txt'
    if (os.path.isfile(filepath) is False):
        print('FILE DOES NOT EXISTS')
        sys.exit(0)
    else:
        file = open(filepath, 'r', encoding='utf-8') 
        w1 = []
        w2 = []
        score = []
    
        for line in file:
            
            data = eval.get_words_for_men_dataset(line)     
            w1.append(data[0])
            w2.append(data[1])
            score.append(data[2])
            #A.append(get_fingerprint_from_image((data[0])))

        if _percentage is not None:
            w1, w2, score = eval.get_percentage_records(w1, w2, score, _percentage)
                    
        df = pd.DataFrame({0: w1, 1: w2, 2: score})
        bunch = Bunch(X=df.values[:, 0:2].astype("object"), y=df.values[:, 2:].astype(np.float) / 5.0)
        print("Sample data from {}: pair \"{}\" and \"{}\" is assigned score {}".format('MEN dataset', bunch.X[0][0], bunch.X[0][1], bunch.y[0]))
        

        A = np.vstack(get_fingerprint_from_image(word, _fingerprints) for word in bunch.X[:, 0])
        B = np.vstack(get_fingerprint_from_image(word, _fingerprints) for word in bunch.X[:, 1])
        
        print (A.shape, B.shape)

        #avs: calculate the cosine distance between the 2 vectores
        # why v1.dot(v2.T): because we are working with matrixes !!!  http://www.thefactmachine.com/cosine-similarity/
        
        scores = _distance_measure(A, B)
        return scipy.stats.spearmanr(scores, bunch.y).correlation

        
def fetch_WS353(_fingerprints, _distance_measure, _percentage):
    
    print ('fetching WS353 dataset')
    filepath = DIR+'/similarity/EN-WS353.txt'
    if (os.path.isfile(filepath) == False):
        print ('FILE DOES NOT EXISTS')
        sys.exit(0)
    else:
        file = open(filepath, 'r', encoding='utf-8') 
        nline = 1
        w1 = []
        w2 = []
        score = []
    
        for line in file:
            if nline != 1:
                data = eval.get_words_for_ws353_dataset(line)
                
                w1.append(data[0])
                w2.append(data[1])
                score.append(data[2])
            nline += 1

            
        df = pd.DataFrame({ 0 : w1, 1 : w2, 2 : score})
        bunch = Bunch(X=df.values[:, 0:2].astype("object"), y=df.values[:, 2:].astype(np.float))
        print("Sample data from {}: pair \"{}\" and \"{}\" is assigned score {}".format('MEN WS353', bunch.X[0][0], bunch.X[0][1], bunch.y[0]))

        A = np.vstack(get_fingerprint_from_image(word, _fingerprints) for word in bunch.X[:, 0])
        B = np.vstack(get_fingerprint_from_image(word, _fingerprints) for word in bunch.X[:, 1])
        
        #print (type(C))
        #print (A.shape, B.shape, C.shape)
        
        #avs: calculate the cosine distance between the 2 vectores
        # why v1.dot(v2.T): because we are working with matrixes !!!  http://www.thefactmachine.com/cosine-similarity/
        scores = _distance_measure(A, B)

        #print (scores[0:5], bunch.y[0:5])
        return scipy.stats.spearmanr(scores, bunch.y).correlation


def fetch_ENTruk(_fingerprints, _distance_measure, _percentage):
    pass


def fetch_ENRG65(_fingerprints, _measure, _percentage):
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

    if _percentage is not None:
        w1, w2, score = eval.get_percentage_records(w1, w2, score, _percentage)

    df = pd.DataFrame({0: w1, 1: w2, 2: score})
    bunch = Bunch(X=df.values[:, 0:2].astype("object"), y=df.values[:, 2:].astype(np.float))

    distance_measure = distances[_measure]
    
    if _measure == 'ssim':
        mode = '2darray'
        A = [get_fingerprint_from_image(word, _fingerprints, mode) for word in bunch.X[:, 0]]
        B = [get_fingerprint_from_image(word, _fingerprints, mode) for word in bunch.X[:, 1]]
    else:
        mode = 'flatten'
        A = np.vstack(get_fingerprint_from_image(word, _fingerprints, mode) for word in bunch.X[:, 0])
        B = np.vstack(get_fingerprint_from_image(word, _fingerprints, mode) for word in bunch.X[:, 1])

    print (len(A))
    scores = distance_measure(A, B)
    print (len(scores))
    print (len(bunch.y))
    return scipy.stats.spearmanr(scores, bunch.y).correlation


def compare_words(_fingerprints, _measure, _words):
    print (_fingerprints, _measure, _words)
    distance_measure = distances[_measure]
    A = np.vstack(get_fingerprint_from_image(_words.split(' ')[0], _fingerprints, 'flatten'))
    B = np.vstack(get_fingerprint_from_image(_words.split(' ')[1], _fingerprints, 'flatten'))

    #print (type(A))
    #print (A)

    dist = distance.cosine(A, B)
    
    C = A*B
    dist2 = 0
    for v1 in C:
        dist2 += v1


    img1 = get_fingerprint_from_image(_words.split(' ')[0], _fingerprints, '2darray')
    img2 = get_fingerprint_from_image(_words.split(' ')[1], _fingerprints, '2darray')
    #print (img1)
    s = ssim(img1, img2)
     
    
    print ("{} distance on words: {} is {}".format(_measure, _words, round(dist, 2)))
    print ("similar bits on words: {} is {}".format(_words, dist2))
    print ("SSIM on words: {} is {}".format(_words, s))

def main(_dataset, _fingerprints, _measure, _percentage):
    
    
    t1=time()

    #distance_measure = distances[_measure]
    
    print ("creating bunch for dataset "+_dataset)
    similarity = datasets[_dataset](_fingerprints, _measure, _percentage)
    db.insert_score(cfg, _fingerprints, _dataset, _measure, similarity)
    print ("Spearman correlation of scores on {}: {}".format(_dataset, similarity))
    
    t2=time()
    #print("\nTime taken for processing snippets\n----------------------------------------------\n{} s".format((t2-t1)))

if __name__ == "__main__":
    
    # Define distance measures
    distances = {
        "cosine": cosine,
        "euclidean": euclidean,
        "similarbits": similarbits,
        "ssim": struc_sim
    }

    # Define datasets
    datasets = {
        "men-dataset": fetch_MEN,
        "WS353-dataset": fetch_WS353,
        "ENTruk-dataset": fetch_ENTruk,
        "EN-RG-65-dataset": fetch_ENRG65
    }
    
    if len(sys.argv) != 6:
        print ("wrong number of arguments")
        print ("python .\process_snippets.py <word or dataset> <fingerprints ID> <distance measure> <percentage> <all or par of words>")
        sys.exit()
    if (sys.argv[5] == 'all'):
        main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))
    else:
        compare_words(sys.argv[2], sys.argv[3], sys.argv[5])

# python evaluate_fingerprints.py men-dataset BSOM_64_1000_305554 cosine 100
# python evaluate_fingerprints.py WS353-dataset BSOM_64_1000_305554 cosine 100

# python evaluate_fingerprints.py EN-RG-65-dataset SDSOM_64_5545_571698_id23 cosine 100 'car automobile'
# python evaluate_fingerprints.py EN-RG-65-dataset SDSOM_64_5545_571698_id23 cosine 100 all
# python evaluate_fingerprints.py EN-RG-65-dataset SDSOM_64_5545_571698_id23 ssim 100 all
# python evaluate_fingerprints.py EN-RG-65-dataset SDSOM_64_5545_571698_id23_size32 cosine 100 'car automobile'
# python evaluate_fingerprints.py EN-RG-65-dataset SDSOM_64_5545_571698_id23_size32 cosine 100 'fruit furnace'
# python evaluate_fingerprints.py EN-RG-65-dataset SDSOM_64_5545_571698_id23_size32 cosine 100 all
# python evaluate_fingerprints.py EN-RG-65-dataset SDSOM_64_5545_571698_id23_size32_sparsify cosine 100 all



     




    



