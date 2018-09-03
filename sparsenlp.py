import os
import sys
import datetime
import pickle
import numpy as np
from sparsenlp.sentencecluster import SentenceCluster
from sparsenlp.sentencesvect import SentenceVect
from sparsenlp.fingerprint import FingerPrint
from sparsenlp.datacleaner import DataCleaner
from sparsenlp.datasets import Datasets

# TODO:
# create GCP VM instance
# install stack
# create fp for EN-RG-65 and EN-WS353


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('USAGE: python {} mode'.format(sys.argv[0]))
        sys.exit(1)

    mode = sys.argv[1]
    
    opts = {
                'id': 31, 
                'paragraph_length': 300, 'dataextension': '3,4', 'n_features': 10000, 'n_components': 700, 'use_idf': False, 'use_hashing': False, 'use_glove': 'glove.6B.50d', 
                #'algorithm': 'KMEANS', 'initialization': True, 'size': 30, 'niterations': 1000, 'minibatch': True, 
                'algorithm': 'MINISOMBATCH', 'initialization': True, 'size': 128, 'niterations': 1000, 'minibatch': True, 
                'testdataset': 'EN-RG-65',
                'verbose': False,
                'repeat': True
        }

    
    if mode == 'tokenize':
        datacleaner = DataCleaner()
        folder = 'F:/RESEARCH/TESE/corpora/wikifiles/01012018/json/enwiki-20180101-pages-articles3.xml-p88445p200507/AE'
        print ('ingesting files')
        datacleaner.ingestfiles(folder, 'pandas')    
        #datacleaner.data = datacleaner.data[:5]
        print ('exploding dataframe_in_snippets')

        datacleaner.explode_dataframe_in_snippets('text', '\n\n+')
        
        time1 = datetime.datetime.now()
        datacleaner.tokenize_pandas_column('text')
        datacleaner.serialize('articles3_AE.bz2', 'F:/RESEARCH/TESE/corpora/wikifiles/01012018/json/')
        time2 = datetime.datetime.now()
        print(time2 - time1)
        
    elif mode == 'create_fps':
        benchmarkdata = FingerPrint(opts, 'numba')
        benchmarkdata.create_fingerprints(fraction=0.5)
    elif mode == 'cluster':

        dataset = Datasets.factory('EN-RG-65')
        words = dataset.get_data('distinct_words')
        print (len(words))

        #dataset = Datasets.factory('EN-WS353')
        #words = dataset.get_data('distinct_words')
        #print (len(words))
        
        vectors = SentenceVect(opts)
        X = vectors.create_vectors()
        snippets_by_word = vectors.create_word_snippets(words)
        
        mycluster = SentenceCluster(opts)
        codebook = mycluster.cluster(X)

        benchmarkdata = FingerPrint(opts, 'numba')
        benchmarkdata.create_fingerprints(snippets_by_word, X, codebook)

        
        sys.exit(0)
  
        
        # sentences vectors
        benchmarkdata = FingerPrint(opts)
        words = benchmarkdata.fetch('EN-RG-65', 'distinct_words')
        #words = 'stove,tumbler'
        #words = 'car,automobile'

        vectors = SentenceVect(opts)
        #snippets_by_word = vectors.create_snippets_by_word(words)
        snippets_by_word, X = vectors.create_vectors(words)
        print (X.shape)

        # codebook creation
        mycluster = SentenceCluster(opts)
        codebook = mycluster.cluster(X)
        print (codebook.shape)
        
        
        print ("Creating Fingerprints...")
        time1 = datetime.datetime.now()
        #benchmarkdata.create_fingerprints(snippets_by_word, X, codebook, words=words)
        benchmarkdata.create_fingerprints(snippets_by_word, X, codebook)
        time2 = datetime.datetime.now()
        print(time2 - time1)
        
        # evaluation
        evaluation_data = benchmarkdata.fetch('data')
        benchmarkdata.evaluate(evaluation_data, 'cosine')
        
    elif mode == 'evaluate':
        benchmarkdata = FingerPrint(opts)
        evaluation_data = benchmarkdata.fetch('data')
        result = benchmarkdata.evaluate(evaluation_data, 'cosine')
        print ('cosine distance {}'.format(result))
        result = benchmarkdata.evaluate(evaluation_data, 'euclidean')
        print ('euclidean distance {}'.format(result))
        result = benchmarkdata.evaluate(evaluation_data, 'similarbits')
        print ('similar bits {}'.format(result))
        result = benchmarkdata.evaluate(evaluation_data, 'structutal similarity')
        print ('structutal similarity {}'.format(result))
        result = benchmarkdata.evaluate(evaluation_data, 'earth movers distance')
        print ('earth movers distance {}'.format(result))

    else:
        raise ValueError('wrong mode !!!')