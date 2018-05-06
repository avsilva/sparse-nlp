
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
from scipy import spatial
from sklearn.datasets.base import Bunch
from PIL import Image

#DIR = '/opt/sparse-nlp/datasets'

DIR = 'C:/Users/andre.silva/web_data/'



#def create_word_fingerprint(_word, _snippets_by_word, _codebook, X, H, W, _sufix):
    #a_original, a_sparse = finger.create_fingerprint(_word, _snippets_by_word, _codebook, X, H, W, _sufix)



def  get_words_for_men_dataset(line):
    
    words = line.split(' ')
    w1 = words[0].split('-')[0]
    w2 = words[1].split('-')[0]
    score = words[2].replace('\n', '')
    return [w1, w2, score]

def get_fingerprint_from_image(_word):
    
    folder = 'BSOM_64_1000_305554'
    filepath = './images/'+folder+'/'+_word+'_BSOM_64_1000_305554.bmp'
    im = Image.open(filepath)
    r, g, b = im.split()
    pix = np.array(r)
    np.place(pix, pix>1, [1])
    pix = pix.flatten()
    return pix

def fetch_MEN():
    print ('fetching MEN dataset')
    filepath = DIR+'/similarity/EN-MEN-LEM.txt'
    if (os.path.isfile(filepath) == False):
        print ('FILE DOES NOT EXISTS')
        sys.exit(0)
    else:
        file = open(filepath, 'r', encoding='utf-8') 
        w1 = []
        w2 = []
        score = []
    
        for line in file:
            
            data = get_words_for_men_dataset(line)     
            w1.append(data[0])
            w2.append(data[1])
            score.append(data[2])
            #A.append(get_fingerprint_from_image((data[0])))
                    
        df = pd.DataFrame({ 0 : w1, 1 : w2, 2 : score})
        bunch = Bunch(X=df.values[:, 0:2].astype("object"), y=df.values[:, 2:].astype(np.float) / 5.0)
        print("Sample data from {}: pair \"{}\" and \"{}\" is assigned score {}".format('MEN dataset', bunch.X[0][0], bunch.X[0][1], bunch.y[0]))
        

        A = np.vstack(get_fingerprint_from_image(word) for word in bunch.X[:, 0])
        B = np.vstack(get_fingerprint_from_image(word) for word in bunch.X[:, 1])
        
        print (A.shape, B.shape)

        #avs: calculate the cosine distance between the 2 vectores
        # why v1.dot(v2.T): because we are working with matrixes !!!  http://www.thefactmachine.com/cosine-similarity/
        scores = np.array([v1.dot(v2.T)/(np.linalg.norm(v1)*np.linalg.norm(v2)) for v1, v2 in zip(A, B)])
        
        return scipy.stats.spearmanr(scores, bunch.y).correlation

        

def fetch_WS353():
    pass

def fetch_SimLex999():
    pass


def main(_name):
    

    # Define datasets
    datasets = {
        "men-dataset": fetch_MEN,
        "WS353-dataset": fetch_WS353,
        "SIMLEX999-dataset": fetch_SimLex999
    }

    t1=time()
    
    print ("creating bunch for dataset "+_name)
    similarity = datasets[_name]()
    print ("Spearman correlation of scores on {} {}".format(_name, similarity))
    

    t2=time()
    print("\nTime taken for processing snippets\n----------------------------------------------\n{} s".format((t2-t1)))

if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        print ("wrong number of arguments")
        print ("python .\process_snippets.py <word or dataset>")
        sys.exit()
    main(sys.argv[1])

#python evaluate_fingerprints.py men-dataset
    




    



