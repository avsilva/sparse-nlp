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
import experiments


# TODO: loop over logs and clean/notclean

def create_fingerprints(opts, snippets_by_word, X, codebook, percentage):
    fingerprints = FingerPrint(opts, opts['engine'])
    fingerprints.create_fingerprints(snippets_by_word, X, codebook, fraction=percentage)


def get_snippets_by_word(datareference):
    dataset = Datasets.factory(datareference)
    words = dataset.get_data('distinct_words')
    snippets_by_word = vectors.create_word_snippets(words)
    return snippets_by_word


def get_vector_representation(opts):
    vectors = SentenceVect(opts)
    X = vectors.create_vectors()
    return X


def get_cluster(opts, X):
    mycluster = SentenceCluster(opts)
    codebook = mycluster.cluster(X)
    return codebook

def clean_files(folder, id):
    files = ['{}X_{}.npz'.format(folder, id), '{}codebook_{}.npy'.format(folder, id)]
    datasets = ['EN-RG-65', 'EN-WS353', 'EN-TRUK', 'EN-SIM999']
    for item in datasets:
        files.append('{}snippets_by_word_{}_{}.pkl'.format(folder, id, item))

    for file in files:
        print (file)
        try:
            os.remove(file)
        except OSError:
            pass



if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('USAGE: python {} mode'.format(sys.argv[0]))
        sys.exit(1)

    mode = sys.argv[1]
    time1 = datetime.datetime.now()

    if mode == 'tokenize':
        datacleaner = DataCleaner()
        folder = '../wikiextractor/jsonfiles/articles3/AB/'
        datacleaner.ingestfiles(folder, 'pandas')   
        datacleaner.explode_dataframe_in_snippets('text', '\n\n+')
        datacleaner.tokenize_pandas_column('text')
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
        opts['repeat'] = False
        if experiments.sentecefolder is not None:
            opts['sentecefolder'] = experiments.sentecefolder
        
        opts['dataset'] = datareference

        vectors = SentenceVect(opts)
        X = vectors.create_vectors()
        snippets_by_word = vectors.create_word_snippets(words)

        mycluster = SentenceCluster(opts)
        codebook = mycluster.cluster(X)

        #fingerprints = FingerPrint(opts, 'ckdtree')

        engine = experiments.engine
        fingerprints = FingerPrint(opts, engine)
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
        #datasets = ['EN-RG-65']
        
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

        for log in experiments.opts:
            if log['id'] == int(id):
                opts = log
        
        assert isinstance(opts, dict), "no log found with id {}".format(id)

        if experiments.sentecefolder is not None:
            opts['sentecefolder'] = experiments.sentecefolder
        opts['repeat'] = True
        vectors = SentenceVect(opts)
        X = vectors.create_vectors()
        
        mycluster = SentenceCluster(opts)
        codebook = mycluster.cluster(X)

    elif mode == 'cluster_fps':

        if len(sys.argv) != 5:
            print('USAGE: python {} logid evaluate datareference percentage'.format(sys.argv[0]))
            sys.exit(1)
        id = sys.argv[2]
        datareference = sys.argv[3]
        percentage = sys.argv[4]
        
        for log in experiments.opts:
            if log['id'] == int(id):
                opts = log

        if not isinstance(opts, dict):
            raise ValueError('no log found with id {}'.format(id))
        
        opts['repeat'] = True
        opts['new_log'] = False
        if experiments.sentecefolder is not None:
            opts['sentecefolder'] = experiments.sentecefolder
        opts['dataset'] = datareference
        opts['engine'] = experiments.engine

        X = get_vector_representation(opts)
        codebook = get_cluster(opts, X)
        snippets_by_word = get_snippets_by_word(datareference)
        create_fingerprints(opts, snippets_by_word, X, codebook, percentage)        

    
    elif mode == 'cluster_fps_all':

        if experiments.sentecefolder is not None:
            folder = experiments.sentecefolder
        else:
            folder = './serializations/'
        datareference = 'EN-RG-65'
        percentage = 1

        for log in experiments.opts:
            id = opts['id']
            opts['repeat'] = True
            opts['new_log'] = False
            opts['dataset'] = datareference
            opts['engine'] = experiments.engine

            X = get_vector_representation(opts)
            codebook = get_cluster(opts, X)
            snippets_by_word = get_snippets_by_word(datareference)
            create_fingerprints(opts, snippets_by_word, X, codebook, percentage) 
            clean_files(folder, id)

    elif mode == 'clean':

        if len(sys.argv) != 3:
            print('USAGE: python {} logid'.format(sys.argv[0]))
            sys.exit(1)

        id = sys.argv[2]

        if experiments.sentecefolder is not None:
            folder = experiments.sentecefolder
        else:
            folder = './serializations/'

        clean_files(folder, id)

    elif mode == 'clean_all':

        if experiments.sentecefolder is not None:
            folder = experiments.sentecefolder
        else:
            folder = './serializations/'

        for log in experiments.opts:
            clean_files(folder, id)


    else:
        raise ValueError('wrong mode !!!')

    time2 = datetime.datetime.now()
    print('time elapsed: {}'.format(time2 - time1))

# python -W ignore sparsenlp.py cluster 201

# python -W ignore sparsenlp.py create_fps 106 EN-RG-65 1

# python sparsenlp.py cluster_fps 101 EN-RG-65 1
# python sparsenlp.py cluster_fps 101 EN-WS353 1

# python -W ignore sparsenlp.py evaluate 3

# python sparsenlp.py clean 999 EN-RG-65

