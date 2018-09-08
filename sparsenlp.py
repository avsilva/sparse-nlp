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
# pass log_id to python sparsenlp.py


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('USAGE: python {} mode'.format(sys.argv[0]))
        sys.exit(1)

    mode = sys.argv[1]
    time1 = datetime.datetime.now()
    
    """
    opts = {
            'id': 2, 
            'paragraph_length': 300, 'dataextension': '3,4', 'n_features': 10000, 'n_components': 700, 'use_idf': False, 'use_hashing': False, 'use_glove': 'glove.6B.100d', 
            #'algorithm': 'KMEANS', 'initialization': True, 'size': 30, 'niterations': 1000, 'minibatch': True, 
            'algorithm': 'MINISOMBATCH', 'initialization': True, 'size': 128, 'niterations': 1000, 'minibatch': True, 
            #'testdataset': 'EN-RG-65',
            'verbose': False,
            'repeat': False
    }
    """
    opts = {'id': 5, 'initialization': True, 'minibatch': False, 'verbose': False, 'n_components': 500, 'size': 100, 
            'paragraph_length': 300, 'niterations': 2000, 'n_features': 10000, 'use_hashing': False,  'use_idf': True, 
            'algorithm': 'MINISOMBATCH', 'use_glove': False, 'dataextension': '3,4'}

    """
    {'id': 16, 'minibatch': True, 'n_components': 700, 'initialization': True, 'n_features': 10000, 
    'use_idf': True, 'algorithm': 'MINISOMBATCH', 'use_hashing': False, 'size': 128,
    'niterations': 1000, 'verbose': False, 'paragraph_length': 300}
    """

    if mode == 'tokenize':
        datacleaner = DataCleaner()
        #folder = 'F:/RESEARCH/TESE/corpora/wikifiles/01012018/json/enwiki-20180101-pages-articles3.xml-p88445p200507/AE'
        folder = '../wikiextractor/jsonfiles/articles3/AB/'
        print ('ingesting files')
        datacleaner.ingestfiles(folder, 'pandas')   
        #datacleaner.data = datacleaner.data[:5]
        print ('exploding dataframe_in_snippets')

        datacleaner.explode_dataframe_in_snippets('text', '\n\n+')
        datacleaner.tokenize_pandas_column('text')
        #datacleaner.serialize('articles3_AE.bz2', 'F:/RESEARCH/TESE/corpora/wikifiles/01012018/json/')
        datacleaner.serialize('articles3_AB.bz2', '../wikiextractor/jsonfiles/')
        
    elif mode == 'create_fps':

        if len(sys.argv) != 4:
            print('USAGE: python {} evaluate dataset'.format(sys.argv[0]))
            sys.exit(1)

        datareference = sys.argv[2]
        percentage = sys.argv[3]

        dataset = Datasets.factory(datareference)
        words = dataset.get_data('distinct_words')
        opts['new_log'] = False
        opts['sentecefolder'] = '/dev/shm/'
        #opts['testdataset'] = 'EN-RG-65'

        vectors = SentenceVect(opts)
        X = vectors.create_vectors()
        snippets_by_word = vectors.create_word_snippets(words)

        mycluster = SentenceCluster(opts)
        codebook = mycluster.cluster(X)

        fingerprints = FingerPrint(opts, 'numba')
        fingerprints.create_fingerprints(snippets_by_word, X, codebook, fraction=percentage)
        

    elif mode == 'evaluate':

        if len(sys.argv) != 3:
            print('USAGE: python {} evaluate dataset'.format(sys.argv[0]))
            sys.exit(1)

        datareference = sys.argv[2]

        dataset = Datasets.factory(datareference)
        evaluation_data = dataset.get_data('data')
        testdataset = list(evaluation_data.values())[0]
        
        opts = {'id': 4, 'algorithm': 'MINISOMBATCH'}

        fingerprints = FingerPrint(opts)
        result = fingerprints.evaluate(evaluation_data, 'cosine')
        print ('result for {}: {}'.format(datareference, result))

    elif mode == 'cluster':

        #dataset = Datasets.factory('EN-RG-65')
        #words = dataset.get_data('distinct_words')
        #print (len(words))

        #dataset = Datasets.factory('EN-WS353')
        #words = dataset.get_data('distinct_words')
        #print (len(words))
        
        opts['sentecefolder'] = '/dev/shm/'
        vectors = SentenceVect(opts)
        X = vectors.create_vectors()
        #snippets_by_word = vectors.create_word_snippets(words)
        
        mycluster = SentenceCluster(opts)
        codebook = mycluster.cluster(X)

        #benchmarkdata = FingerPrint(opts, 'numba')
        #benchmarkdata.create_fingerprints(snippets_by_word, X, codebook)
        
    elif mode == 'evaluate_all':
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

    time2 = datetime.datetime.now()
    print('time elapsed: {}'.format(time2 - time1))

# python sparsenlp.py cluster 

# python sparsenlp.py create_fps  EN-RG-65 1
# python sparsenlp.py create_fps  EN-WS353 0.5

# python sparsenlp.py evaluate EN-RG-65
# python sparsenlp.py evaluate EN-WS353