import os
import sys
import ast
import datetime
import pickle
import numpy as np
from sparsenlp.sentencecluster import SentenceCluster
from sparsenlp.sentencesvect import SentenceVect
from sparsenlp.fingerprint import FingerPrint
from sparsenlp.datacleaner import DataCleaner
from sparsenlp.datasets import Datasets


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('USAGE: python {} mode'.format(sys.argv[0]))
        sys.exit(1)

    mode = sys.argv[1]
    time1 = datetime.datetime.now()

    opts = [
        {'id': 201, 'initialization': True, 'minibatch': True, 'verbose': False, 'n_components': 700, 
    'size': 64, 'paragraph_length': 400, 'niterations': 1000, 'n_features': 10000, 'use_hashing': False, 'use_idf': True, 
    'algorithm': 'KMEANS', 'use_glove': False, 'dataextension': '', 'token': 'text'},
    ]



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

        if len(sys.argv) != 5:
            print('USAGE: python {} logid evaluate dataset'.format(sys.argv[0]))
            sys.exit(1)

        id = sys.argv[2]
        datareference = sys.argv[3]
        percentage = sys.argv[4]

        with open('./logs/log_{}'.format(id), 'r', encoding='utf-8') as handle:
            datafile = handle.readlines()
            for x in datafile:
                log = ast.literal_eval(x)

        opts = log

        dataset = Datasets.factory(datareference)
        words = dataset.get_data('distinct_words')

        opts['new_log'] = False
        opts['sentecefolder'] = '/dev/shm/'
        opts['dataset'] = datareference

        vectors = SentenceVect(opts)
        X = vectors.create_vectors()
        snippets_by_word = vectors.create_word_snippets(words)

        mycluster = SentenceCluster(opts)
        codebook = mycluster.cluster(X)

        #fingerprints = FingerPrint(opts, 'ckdtree')
        fingerprints = FingerPrint(opts, 'numba')
        fingerprints.create_fingerprints(snippets_by_word, X, codebook, fraction=percentage)
        

    elif mode == 'evaluate':

        if len(sys.argv) != 3:
            print('USAGE: python {} evaluate log'.format(sys.argv[0]))
            sys.exit(1)

        id = sys.argv[2]
        with open('./logs/log_{}'.format(id), 'r', encoding='utf-8') as handle:
            datafile = handle.readlines()
            for x in datafile:
                log = ast.literal_eval(x)
        
        opts = {'id': id, 'algorithm': log['algorithm']}

        fingerprints = FingerPrint(opts)

        datasets = ['EN-RG-65', 'EN-WS353', 'EN-TRUK', 'EN-SIM999']
        #datasets = ['EN-WS353']
        
        #metrics = ['cosine']
        metrics = ['cosine', 'euclidean', 'similarbits', 'structutal similarity', 'earth movers distance']

        best_scores = []
        for dataset in datasets:
            best = 0
            testdata = Datasets.factory(dataset)
            
            evaluation_data = testdata.get_data('data')
            
            for metric in metrics:
                result = fingerprints.evaluate(evaluation_data, metric)
                
                if result['score'] > best:
                    #best_scores['dataset'] = [metric.upper(), result['score'], str(result['percentage']) + ' %']
                    best_scores.append ([dataset, metric.upper(), result['score'], str(result['percentage']) + ' %'])
                    best = result['score']
                #print ('\nresult for {} {}: {}'.format(metric, dataset, result))

        for best in best_scores:
            print(best)

    elif mode == 'cluster':
        
        if len(sys.argv) != 3:
            print('USAGE: python {} logid '.format(sys.argv[0]))
            sys.exit(1)
        id = sys.argv[2]

        for log in opts:
            if log['id'] == int(id):
                opts = log
        
        assert isinstance(opts, dict), "no log found with id {}".format(id)

        # opts['sentecefolder'] = '/dev/shm/'
        opts['repeat'] = True
        vectors = SentenceVect(opts)
        X = vectors.create_vectors()
        
        mycluster = SentenceCluster(opts)
        codebook = mycluster.cluster(X)

    elif mode == 'cluster_fps':

        if len(sys.argv) != 5:
            print('USAGE: python {} logid evaluate dataset'.format(sys.argv[0]))
            sys.exit(1)
        id = sys.argv[2]
        datareference = sys.argv[3]
        percentage = sys.argv[4]
        
        for log in opts:
            if log['id'] == int(id):
                opts = log

        if not isinstance(opts, dict):
            raise ValueError('no log found with id {}'.format(id))
        
        opts['new_log'] = False
        opts['sentecefolder'] = '/dev/shm/'
        opts['dataset'] = datareference

        vectors = SentenceVect(opts)
        X = vectors.create_vectors()
        
        mycluster = SentenceCluster(opts)
        codebook = mycluster.cluster(X)

        dataset = Datasets.factory(datareference)
        words = dataset.get_data('distinct_words')
        snippets_by_word = vectors.create_word_snippets(words)

        fingerprints = FingerPrint(opts, 'numba')
        fingerprints.create_fingerprints(snippets_by_word, X, codebook, fraction=percentage)

    elif mode == 'clean':

        if len(sys.argv) != 4:
            print('USAGE: python {} logid dataset'.format(sys.argv[0]))
            sys.exit(1)

        id = sys.argv[2]
        datareference = sys.argv[3]
        files = ['./serializations/X_{}.npz'.format(id), './serializations/codebook_{}.npy'.format(id)]

        for file in files:
            print (file)
            try:
                os.remove(file)
            except OSError:
                pass

    else:
        raise ValueError('wrong mode !!!')

    time2 = datetime.datetime.now()
    print('time elapsed: {}'.format(time2 - time1))

# python -W ignore sparsenlp.py cluster 201

# python sparsenlp.py create_fps 106 EN-RG-65 1

# python sparsenlp.py cluster_fps 101 EN-RG-65 1
# python sparsenlp.py cluster_fps 101 EN-WS353 1

# python -W ignore sparsenlp.py evaluate 3

# python sparsenlp.py clean 999 EN-RG-65

