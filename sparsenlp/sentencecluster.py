import os
import sys
import pandas as pd
import numpy
import pickle
from minisom import MiniSom
import sparsenlp.sentencesvect as vect
import sparsenlp.modelresults as modelres
import utils.database as db
import utils.decorators as decorate
from sklearn.cluster import KMeans, MiniBatchKMeans


class SentenceCluster():
    """Initializes an instance of sentences obtained from wikipedia pre-processed paragraphs.

        The instance is intended for further processing including vectorization and clustering

        Attributes
        ----------
        opts : dict
            instance settings (e.g paragraph_length, size, algorithm)
        path: str
            folder path where serialization files are stored 

        Methods
        -------
        serialize_sentences()
            Get sentences from database and serialize to file
        create_sentence_vector()
            Creates vector representation of sentences

    """

    path = './serializations/'
    
    
    def __init__(self, opts):
        """
        Parameters
        ----------
        opts : dict
            instance settings (e.g paragraph_length, size, algorithm)
        """

        self.opts = opts
        self.__train_data = None
        self.X = None
        self.algos = {'KMEANS': self._kmeans, 'MINISOMBATCH': self._minisombatch,
                      'MINISOMRANDOM': self._minisomrandom}

    def serialize_sentences(self):
        """Serializes text sentences from database
        
        Returns
        -----------
        pandas dataframe
        """

        path = '{}sentences/'.format(self.path)
        filepath = '{}{}.bz2'.format(path, self.opts['paragraph_length'])
        if (os.path.isfile(filepath) is True):
            self.__train_data = self._read_serialized_sentences_text()
        else:
            df_train = db.get_cleaned_data(None, self.opts['paragraph_length'])
            self.__train_data = df_train
            self._serialize_sentences_text('{}sentences/'.format(self.path))

        return self.__train_data
        
    def _serialize_sentences_text(self, path):
        """Serializes pandas dataframe text sentences"""

        self.__train_data.to_pickle('{}{}.bz2'.format(path, 
                                    self.opts['paragraph_length']), 
                                    compression="bz2")
    
    """
    @decorate.elapsedtime_log
    def create_sentence_vector(self):
        

        filepath = './serializations/X_{}.npz'.format(self.opts['id'])
        if (os.path.isfile(filepath) is True):
            self.__X = self._read_serialized_sentences_vector()
        else:
            
            vectors = vect.SentenceVect(self.opts)
            results = modelres.ModelResults('./logs')
            #print(results.get_results()) 
            result_id = vectors.check_same_sentence_vector(results.get_results())
            if result_id is not False:
                print ('Using existing vector representation: id {}'.format(result_id))
                with open('{}X_{}.npz'.format(self.path, result_id), 'rb') as handle:
                    self.__X = pickle.load(handle)
            else:
                data = self._read_serialized_sentences_text()
                self.__X = vectors.sentence_representation(data.cleaned_text)
                self._serialize_sentence_vector()
        return self.__X
    """

    def _read_serialized_sentences_vector(self):
        """Returns sparse matrix of sentence vectors"""
        
        with open('{}X_{}.npz'.format(self.path, self.opts['id']), 
                  'rb') as handle:
            self.__X = pickle.load(handle)
        return self.__X

    @decorate.elapsedtime_log
    def cluster(self, X):
        """Clusters sentence vectors using the instance algorithm"""

        
        logs = modelres.ModelResults('./logs')
        results = logs.get_results(exception=self.opts['id'])
        same_codebook = self.check_same_codebook(results)

        if len(same_codebook) > 0:
            log_id = min(same_codebook)
            print('Using existing codebook: id {}'.format(log_id))
            with open('./serializations/codebook_{}.npy'.format(log_id), 'rb') as handle:
                codebook = pickle.load(handle)
        
        else:
            print('Creating new codebook: id {}'.format(self.opts['id']))
            self.X = X
            # dict self.algos contains mapping of algorithm to clustering method
            codebook = self.algos[self.opts['algorithm']]()

            with open('./serializations/codebook_{}.npy'.format(self.opts['id']),
                    'wb') as handle:
                pickle.dump(codebook, handle)
       
        return codebook

    def check_same_codebook(self, results):
        
        keys = ['paragraph_length', 'n_features', 'n_components', 'use_idf', 
                'use_hashing', 'algorithm', 'initialization', 'size', 
                'niterations', 'minibatch']

        same_codebooks = []
        for result in results:
            
            equal = True
            for key in keys:
                if result[key] != self.opts[key]:
                    equal = False
                    continue

            if equal is True:
                #return result['id']
                same_codebooks.append(result['id'])
            
        return same_codebooks

    def _kmeans(self):
        """Clusters sentence vectors using kmeans algorithm
        
        Returns
        -------
        numpy ndarray
            labels for each vector representation
        """
        size = self.opts['size'] * self.opts['size']
        if self.opts['minibatch'] is True:
            labels = self.create_kmeans_minibatch_cluster(size)
        else:
            labels = self.create_kmeans_cluster(size)
        
        return labels

    def _minisombatch(self):
        """Clusters sentence vectors using minisombatch algorithm
        
        Returns
        -------
        numpy ndarray
            codebook (weights) of the trained SOM
        """

        H = int(self.opts['size'])
        W = int(self.opts['size'])
        N = self.X.shape[1]
        som = MiniSom(H, W, N, sigma=1.0, random_seed=1)
        if self.opts['initialization']:
            som.random_weights_init(self.X)
        som.train_batch(self.X, self.opts['niterations'])
        return som.get_weights()

    def _minisomrandom(self):
        """Clusters sentence vectors using minisomrandom algorithm
        
        Returns
        -------
        numpy ndarray
            codebook (weights) of the trained SOM
        """

        H = int(self.opts['size'])
        W = int(self.opts['size'])
        N = self.X.shape[1]
        som = MiniSom(H, W, N, sigma=1.0, random_seed=1)
        if self.opts['initialization']:
            som.random_weights_init(self.X)
        som.train_random(self.X, self.opts['niterations'])
        return som.get_weights()

    def create_kmeans_minibatch_cluster(self, size):
        
        km = MiniBatchKMeans(n_clusters=size, init='k-means++', n_init=1,
                             init_size=1000, batch_size=1000, 
                             verbose=self.opts['verbose'])
        km.fit(self.X)
        return km.labels_

    def create_kmeans_cluster(self, size):
        
        km = KMeans(n_clusters=size, init='k-means++', max_iter=100, n_init=1,
                    verbose=self.opts['verbose'])
        km.fit(self.X)
        return km.labels_
