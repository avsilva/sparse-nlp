import os
import sys
import ast
import datetime
import pickle
import numpy as np
import pandas as pd
from sparsenlp.sentencecluster import SentenceCluster
from sparsenlp.sentencesvect import SentenceVect
from sparsenlp.fingerprint import FingerPrint
from sparsenlp.datacleaner import DataCleaner
from sparsenlp.datasets import Datasets
import experiments
import msgpack
import gc
import collections
from scipy.spatial import distance
from sklearn.metrics import pairwise_distances
from scipy.sparse import csr_matrix
from scipy import sparse
from nltk.corpus import stopwords

from web.datasets.categorization import fetch_AP, fetch_battig, fetch_BLESS, fetch_ESSLI_1a, fetch_ESSLI_2b, fetch_ESSLI_2c
from web.datasets.similarity import fetch_MEN, fetch_WS353, fetch_SimLex999, fetch_RW, fetch_multilingual_SimLex999, fetch_RG65, fetch_MTurk
from web.analogy import *
from six import iteritems
from web.embeddings import fetch_GloVe, fetch_FastText, fetch_Mine
from web.evaluate import evaluate_categorization, evaluate_similarity, evaluate_analogy, evaluate_on_semeval_2012_2


# https://www.benfrederickson.com/dont-pickle-your-data/
# TODO: loop over logs and clean/notclean

def create_fingerprints(opts, snippets_by_word, X, codebook, sparsity):
    fingerprints = FingerPrint(opts, opts['engine'])
    fingerprints.create_fingerprints(snippets_by_word, X, codebook, sparsity)


def get_snippets_by_word(vectors, datareference):
    dataset = Datasets.factory(datareference)
    words = dataset.get_data('distinct_words')
    snippets_by_word = vectors.create_word_snippets(words)
    return snippets_by_word


def get_vector_representation(vectors):
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

def get_words(datasets=None):
    words = []
    if datasets is None:
        datasets = ['EN-WS353', 'EN-RG-65', 'EN-TRUK', 'EN-SIM999']
    for d in datasets:
        dataset = Datasets.factory(d)
        distinct_words = dataset.get_data('distinct_words')[d]
        words = list(set(distinct_words)) + words
    
    words = [w.lower() for w in words]
    words = list(set(words))
    print('{} distict words in {} datasets'.format(len(words), len(datasets)))
    return words

def load_msgpack_gc(_file):
    output = open(_file, 'rb')

    # disable garbage collector
    gc.disable()

    mydict = msgpack.unpack(output)

    # enable garbage collector again
    gc.enable()
    output.close()
    return mydict

def load_pickle_gc(_file):
    output = open(_file, 'rb')
    # disable garbage collector
    #gc.disable()
    mydict = pickle.load(output)
    # enable garbage collector again
    #gc.enable()
    output.close()
    return mydict

def gimme_glove(_dim):
    with open("./embeddings/glove.6B/glove.6B.{}d.txt".format(_dim), encoding='utf-8') as glove_raw:
        for line in glove_raw.readlines():
            splitted = line.split(' ')
            yield splitted[0], np.array(splitted[1:], dtype=np.float)


FILES = [
            './serializations2/snippetsbyword_all_datasets_1234_text_300_0.pkl', 
            './serializations2/snippetsbyword_all_datasets_1234_text_300_1.pkl',
            './serializations2/snippetsbyword_all_datasets_1234_text_300_2.pkl', 
            './serializations2/snippetsbyword_all_datasets_1234_text_300_3.pkl',
            './serializations2/snippetsbyword_all_datasets_1234_text_300_4.pkl', 
            './serializations2/snippetsbyword_all_datasets_1234_text_300_5.pkl'
        ]

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

    elif mode == 'calculate_freqs':

        paragraph_length = 300
        dataextensions = '1234'
        column = 'text'
        stopWords = set(stopwords.words('english'))
        

        counts_by_word = {}
        counter = collections.Counter()
        
        for file in FILES:
            print ('Loading {} '.format(file))
            mydict = load_pickle_gc(file)
  
            for word in stopWords:
                try:
                    del mydict[word]
                except KeyError as e:
                    pass

            i = 0
            for word, values in mydict.items():
                
                counts = sum ([value['counts'] for value in values])
                try:
                    counts_by_word[word] += counts 
                    counter[word] += counts
                except KeyError as e:
                    try:
                        counts_by_word[word] = counts
                        counter[word] += counts
                    except KeyError as e:
                        pass
            del mydict
        print (counts_by_word['car'])
        print (counter.most_common(10))
        with open('./serializations2/vocabulary_{}_{}_{}.pkl'.format(dataextensions, column, paragraph_length), 'wb') as f:
            pickle.dump(counter, f)

        del counter['idx']
        wordcounts =  list(counter.values())
        print('There are {} words not considering stop words'.format(sum(wordcounts)))
    
    elif mode == 'csr_matrix':

        id = 100031
        filepath = './images/fp_{}/dict_{}.npy'.format(id, id)
        with open(filepath, 'rb') as handle:
            kmeans_fp = pickle.load(handle)
        print ('{} vocab in dict_{}.npy with dimensionality {}'.format(len(list(kmeans_fp.keys())), id, len(kmeans_fp['car'])))
        
        word1 = kmeans_fp['car']
        word2 = kmeans_fp['automobile']
        word_sim = 1 - distance.cosine(word1, word2)
        print('word sim {} '.format(word_sim))

        vocabulary = np.asarray(list(kmeans_fp.keys()))
        
        index1 = np.where(vocabulary=='car')
        index2 = np.where(vocabulary=='automobile')
        S = csr_matrix(list(kmeans_fp.values()))

        #sparse.save_npz("./S.npz", S)
        #your_matrix_back = sparse.load_npz("./S.npz")
        
        word1 = S[index1]
        word2 = S[index2]
        print (word1.shape)
        #print (word2)
        word_sim = 1 - distance.cosine(word1.toarray(), word2.toarray())
        print('word sim {} '.format(word_sim))

        D = pairwise_distances(S, word1.reshape(1, -1), metric='cosine')
        most_similar_words = [vocabulary[id] for id in D.argsort(axis=0).flatten()[0:5]]
        print(most_similar_words)

        word = kmeans_fp['car']
        words = ['bangkok is to thailand as havana is to ?']

        vocabulary = np.asarray(list(kmeans_fp.keys()))
        vectors = np.asarray(list(kmeans_fp.values()))
        D = pairwise_distances(vectors, word.reshape(1, -1), metric='cosine')
        #index = D.argsort(axis=0).flatten()[0:3]
        most_similar_words = [vocabulary[id] for id in D.argsort(axis=0).flatten()[0:5]]
        print(most_similar_words)
    
    elif mode == 'calculate_vocab':

        id = 100031
        filepath = './images/fp_{}/dict_{}.npy'.format(id, id)
        with open(filepath, 'rb') as handle:
            kmeans_fp = pickle.load(handle)
        print ('{} vocab in dict_{}.npy with dimensionality {}'.format(len(list(kmeans_fp.keys())), id, len(kmeans_fp['car'])))
        

        dim = 50
        glove = {w: x for w, x in gimme_glove(dim)}
        print ('{} vocab in glove {} with dimensionality {}'.format(len(list(glove.keys())), dim, len(glove['car'])))

        vocab = []
        for file in FILES:
            print ('Loading {} '.format(file))
            mydict = load_pickle_gc(file)
            vocab += list(mydict.keys())
            print ('{} vocab in {} '.format(len(list(mydict.keys())), file))
            del mydict
        vocab2 = set(vocab)
        print ('{} vocab in dictionary '.format(len(vocab2)))

    elif mode == 'evaluate_all':

        if len(sys.argv) != 3:
            print('USAGE: python {} evaluate_all logid'.format(sys.argv[0]))
            sys.exit(1)
        id = sys.argv[2]

        #w_embedding = fetch_Mine(id, format="dict", normalize=False, lower=False, clean_words=False)
        w_embedding = fetch_Mine(id, format="csr", normalize=False, lower=False, clean_words=False)

        
        print('{}'.format(' '.join(['-' for x in range(30)])))
        # SIMILARITY
        similarity_results = {}
        similarity_tasks = {
            #"RG65": fetch_RG65(),
            #"MEN": fetch_MEN(),    
            #"WS353": fetch_WS353(),
            #"WS353R": fetch_WS353(which="relatedness"),
            #"WS353S": fetch_WS353(which="similarity"),
            #"SimLex999": fetch_SimLex999(),
            #"RW": fetch_RW(),
            #"MTurk": fetch_MTurk(),
            #"multilingual_SimLex999": fetch_multilingual_SimLex999()
        }
        for name, data in similarity_tasks.items():
            similarity_results[name] = evaluate_similarity(w_embedding, data.X, data.y)
            print ("Spearman correlation of scores on {} {}".format(name, similarity_results[name]))

        print('{}'.format(' '.join(['-' for x in range(30)])))

        # ANALOGY
        analogy_tasks = {
            "Google": fetch_google_analogy(),
            #"MSR": fetch_msr_analogy()
        }
        analogy_results = {}
        for name, data in analogy_tasks.items():
            analogy_results[name] = evaluate_analogy(w_embedding, data.X, data.y)
            print("Analogy prediction accuracy on {} {}".format(name, analogy_results[name]))
        analogy_results["SemEval2012_2"] = evaluate_on_semeval_2012_2(w_embedding)['all']
        print("Analogy prediction accuracy on {} {}".format("SemEval2012", analogy_results["SemEval2012_2"]))


        print('{}'.format(' '.join(['-' for x in range(30)])))
        # CATEGORIZATION
        categorization_tasks = {
                #"AP": fetch_AP(),
                #"BLESS": fetch_BLESS(),
                #"Battig": fetch_battig(),
                #"ESSLI_2c": fetch_ESSLI_2c(),
                #"ESSLI_2b": fetch_ESSLI_2b(),
                #"ESSLI_1a": fetch_ESSLI_1a()
        }
        categorization_results = {}
        for name, data in categorization_tasks.items():
            categorization_results[name] = evaluate_categorization(w_embedding, data.X, data.y)
            print("Cluster purity on {} {}".format(name, categorization_results[name]))
        
    elif mode == 'create_kmeans_fps':

        if len(sys.argv) != 4:
            print('USAGE: python {} create_kmeans_fps logid sparsity'.format(sys.argv[0]))
            sys.exit(1)

        #id = 100031
        id = sys.argv[2]
        sparsity = sys.argv[3]
        paragraph_length = 300
        dataextensions = '12345'
        column = 'text'

        with open('./logs/log_{}'.format(id), 'r', encoding='utf-8') as handle:
            datafile = handle.readlines()
            for x in datafile:
                log = ast.literal_eval(x)
        opts = log

        
        mode = 'most_common'
        top = 10000
        if mode == 'most_common':
            with open('./vocabulary_{}_{}_{}.pkl'.format(dataextensions, column, paragraph_length), 'rb') as f:
                Counter = pickle.load(f)

            words= [word for word, word_count in Counter.most_common(top)]
            if 'idx' in words:
                i = words.index("idx")
                del words[i]
        
        elif mode == 'datatsets':
            #words = get_words(['EN-RG-65', 'EN-WS353', 'EN-TRUK', 'EN-SIM999', 'EN-MEN-LEM'])
            words = get_words(['EN-RG-65'])
            print ('Words {}'.format(len(words)))
        
        mode2 = ''
        if mode2 == 'append':
            filepath = './images/fp_{}/dict_{}.npy'.format(id, id)
            try:
                with open(filepath, 'rb') as handle:
                    kmeans_fp = pickle.load(handle)
                print ('{} vocab in dict_{}.npy with dimensionality {}'.format(len(list(kmeans_fp.keys())), id, len(kmeans_fp['car'])))
            except FileNotFoundError as e:
                kmeans_fp = {}

            words2 = []
            for w in words:
                if w not in kmeans_fp.keys():
                    words2.append(w)
            print ('New words to append {}'.format(len(words2)))
            print ('First word to append {}'.format(words2[2]))
        
        snippets_by_word = {}
        for file in FILES:
            print ('Loading {} '.format(file))
            mydict = load_pickle_gc(file)
            
            for word in words:
                #print ('Word id {} '.format(word))
                try:
                    snippets_by_word[word] += mydict[word] 
                except KeyError as e:
                    try:
                        snippets_by_word[word] = mydict[word]
                    except KeyError as e:
                        pass

            #word_snippets[word] += mydict[word] 
            del mydict

        #print ('Word car appears in {} documents '.format(len(snippets_by_word['car'])))

        opts['new_log'] = False
        opts['repeat'] = False
        if experiments.sentecefolder is not None:
            opts['sentecefolder'] = experiments.sentecefolder
        #opts['dataset'] = datareference

        #vectors = SentenceVect(opts)
        #X = vectors.create_vectors()

        mycluster = SentenceCluster(opts)
        codebook = mycluster.cluster(None)

        engine = experiments.engine
        fingerprints = FingerPrint(opts, engine)
        fingerprints.create_fingerprints(snippets_by_word, None, codebook, sparsity)
    
    elif mode == 'read_chunks':
        
        word = 'car'
        word_snippets = {}
        word_snippets[word] = []
        for file in FILES:
            mydict = load_pickle_gc(file)
            #mydict = load_msgpack_gc(file)
            print (len(mydict[word]))
            word_snippets[word] += mydict[word] 
            del mydict
        print (len(word_snippets[word]))
    
    elif mode == 'create_chunks':
        
        datacleaner = DataCleaner()
        paragraph_length = 300
        dataextensions = '12345'
        column = 'text'

        if experiments.sentecefolder is not None:
            basefolder = experiments.sentecefolder
        else:
            basefolder = './'
        with open('{}counter_all_datasets_{}_text_{}.pkl'.format(basefolder, dataextensions, paragraph_length), 'rb') as f:
            counter = pickle.load(f)
        
        rows = len(counter)
        print(rows)
        chunksize = 200000
        num_chunks = (rows + chunksize - 1) // chunksize
        print (num_chunks)
   
        for i in range(num_chunks): 
            if i != 0:
                init = (i*chunksize)+1
                final = (i+1)*chunksize
                print(init, final)
            else:
                init = i*chunksize
                final = (i+1)*chunksize
            print(init, final)
            counter1 = counter[init:final]
            snippets_by_word = datacleaner.get_word_snippets2(counter1)
            del counter1
            #with open('./snippetsbyword_all_datasets_1234_{}_{}_{}.msgpack'.format(column, paragraph_length, i), 'wb') as f:
            #    msgpack.pack(snippets_by_word, f)
            with open('{}snippetsbyword_all_datasets_{}_{}_{}_{}.pkl'.format(basefolder, dataextensions, column, paragraph_length, i), 'wb') as f:
                pickle.dump(snippets_by_word, f)
            del snippets_by_word
        
        #counter = counter[:200000]
        print (type(counter))
        print (len(counter))
        print (counter[5])
    
    elif mode == 'create_counter':
        paragraph_length = 300
        dataextensions = '12345'
        column = 'text'

        if experiments.sentecefolder is not None:
            basefolder = experiments.sentecefolder
        else:
            basefolder = './'
        
        dataframe = pd.read_pickle('{}dataframe_{}_text_{}.pkl'.format(basefolder, dataextensions, paragraph_length), compression='bz2')
        datacleaner = DataCleaner()
        counter = datacleaner.get_counter(dataframe, column)
        with open('{}counter_all_datasets_{}_text_{}.pkl'.format(basefolder, dataextensions, paragraph_length), 'wb') as f:
            pickle.dump(counter, f)
    
    elif mode == 'create_dataframe_snippets':
        paragraph_length = 300
        dataextensions = '3,4,5'
        column = 'text'
        datacleaner = DataCleaner()
        opts = {'id': 0, 'paragraph_length': paragraph_length, 'dataextension': dataextensions}
        if experiments.sentecefolder is not None:
            opts['sentecefolder'] = experiments.sentecefolder
        
        vectors = SentenceVect(opts)
        dataframe = vectors._read_serialized_sentences_text()
        dataframe = dataframe[['text']]
        dataframe = datacleaner.tokenize_text(dataframe, column)
        print(dataframe.shape)
        print(dataframe.columns)
        print(dataframe['text'][100])
        ext = '12'+str(dataextensions.replace(',', ''))
        if experiments.sentecefolder is not None:
            dataframe.to_pickle('{}dataframe_{}_text_{}.pkl'.format(experiments.sentecefolder, ext, paragraph_length), compression='bz2')
        else:
            dataframe.to_pickle('./dataframe_{}_text_{}.pkl'.format(ext, paragraph_length), compression='bz2')

        
    
    elif mode == 'create_fps2':

        if len(sys.argv) != 4:
            print('USAGE: python {} logid word_snippets'.format(sys.argv[0]))
            sys.exit(1)

        id = sys.argv[2]
        wordsnippets = sys.argv[3]
        sparsity = 0
        
        with open('./logs/log_{}'.format(id), 'r', encoding='utf-8') as handle:
            datafile = handle.readlines()
            for x in datafile:
                log = ast.literal_eval(x)
        opts = log
        opts['new_log'] = False
        opts['repeat'] = False
        if experiments.sentecefolder is not None:
            opts['sentecefolder'] = experiments.sentecefolder
        #opts['dataset'] = datareference

        vectors = SentenceVect(opts)
        X = vectors.create_vectors()
        snippets_by_word = vectors.get_word_snippets(wordsnippets)

        mycluster = SentenceCluster(opts)
        codebook = mycluster.cluster(X)
        
        engine = experiments.engine
        fingerprints = FingerPrint(opts, engine)
        fingerprints.create_fingerprints(snippets_by_word, X, codebook, sparsity)
    
    elif mode == 'create_fps':

        if len(sys.argv) != 4:
            print('USAGE: python {} logid dataset'.format(sys.argv[0]))
            sys.exit(1)

        id = sys.argv[2]
        datareference = sys.argv[3]
        sparsity = 0

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

        engine = experiments.engine
        fingerprints = FingerPrint(opts, engine)
        fingerprints.create_fingerprints(snippets_by_word, X, codebook, sparsity)
        

    elif mode == 'evaluate':

        if len(sys.argv) != 4:
            print('USAGE: python {} evaluate logid sparsity'.format(sys.argv[0]))
            sys.exit(1)

        id = sys.argv[2]
        sparsity = float(sys.argv[3])
        with open('./logs/log_{}'.format(id), 'r', encoding='utf-8') as handle:
            datafile = handle.readlines()
            for x in datafile:
                log = ast.literal_eval(x)
        
        opts = {'id': id, 'algorithm': log['algorithm']}

        fingerprints = FingerPrint(opts)

        #datasets = ['EN-RG-65', 'EN-WS353', 'EN-WSR353R', 'EN-WSS353S', 'EN-TRUK', 'EN-SIM999', 'EN-MEN-LEM', 'EN-RW']
        datasets = ['EN-RG-65']
        
        metrics = ['cosine']
        #metrics = ['cosine', 'euclidean', 'similarbits', 'structutal similarity', 'earth movers distance']

        best_scores = []
        for dataset in datasets:
            best = 0
            testdata = Datasets.factory(dataset)
            
            evaluation_data = testdata.get_data('data')
            #print (evaluation_data)
            """
            d = datasets[0]
            start = 2
            limit = 6
            a = evaluation_data[d][0][start:limit]
            b = evaluation_data[d][1][start:limit]
            c = evaluation_data[d][2][start:limit]
            print(a)
            print(b)
            print(c)
            evaluation_data = {d: [a, b, c]}
            """
            
            for metric in metrics:
                result = fingerprints.evaluate(evaluation_data, metric, sparsity)
                print (result)
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

        if len(sys.argv) != 3:
            print('USAGE: python {} logid'.format(sys.argv[0]))
            sys.exit(1)
        id = sys.argv[2]
        datareference = 'EN-RG-65'
        sparsity = 0
        
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

        vectors = SentenceVect(opts)
        X = get_vector_representation(vectors)
        codebook = get_cluster(opts, X)
        snippets_by_word = get_snippets_by_word(vectors, datareference)
        create_fingerprints(opts, snippets_by_word, X, codebook, sparsity)        

    
    elif mode == 'cluster_fps_all':

        if experiments.sentecefolder is not None:
            folder = experiments.sentecefolder
        else:
            folder = './serializations/'


        datareference = 'EN-RG-65'
        sparsity = 0

        for log in experiments.opts:
            id = log['id']
            log['repeat'] = True
            log['new_log'] = False
            log['dataset'] = datareference
            log['engine'] = experiments.engine
            log['sentecefolder'] = folder

            vectors = SentenceVect(log)
            X = get_vector_representation(vectors)
            codebook = get_cluster(log, X)
            snippets_by_word = get_snippets_by_word(vectors, datareference)
            create_fingerprints(log, snippets_by_word, X, codebook, sparsity) 
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
# python -W ignore sparsenlp.py create_fps 2001 EN-RG-65
# python -W ignore sparsenlp.py create_fps2 100031 snippetsbyword_all_datasets_1234_text_300.pkl
# python sparsenlp.py cluster_fps_all
# python -W ignore sparsenlp.py evaluate 3
# python sparsenlp.py clean 999 EN-RG-65
# python -W ignore sparsenlp.py create_kmeans_fps 100031

#! python3 sparsenlp.py create_dataframe_snippets
#! python3 sparsenlp.py create_counter #1:11:10
#! python3 sparsenlp.py create_chunks #0:08:41
#%time! python3 sparsenlp.py read_chunks