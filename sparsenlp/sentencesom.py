import os
import sys
import pandas as pd
import numpy
import pickle
from minisom import MiniSom
import sentencesvect as vect
import testdataset
import fingerprint as fp
sys.path.append(os.path.abspath('./utils'))
import database as db
import decorators as decorate


class SentenceSom():
    
    @decorate.create_log
    def __init__(self, opts):
        """Initializes a Sentence Self Organizing Maps.

        """
        self.opts = opts
        self.som_size = opts['size']
        self.sentece_length = opts['paragraph_length']
        self.train_data = None
        self.X = None

    def get_sentences(self, sentece_length=None):
        """Get sentences from database"""

        if sentece_length is None:
            sentece_length = self.sentece_length

        df_train = db.get_cleaned_data(None, sentece_length)
        self.train_data = df_train.cleaned_text
        return self.train_data

    def serialize_sentences_text(self, path):
        """Serializes dataframe text sentences"""
        self.train_data.to_pickle('{}{}.bz2'.format(path, self.sentece_length), compression="bz2")
        
    def read_serialized_sentences_text(self, path):
        """Loads Serialized dataframe text sentences"""
        
        self.train_data = pd.read_pickle('{}{}.bz2'.format(path, self.sentece_length), compression="bz2")
        return self.train_data
    
    @decorate.elapsedtime_log
    def create_sentence_vector(self, data):
        """Creates vector representation of sentences"""
        vectors = vect.SentenceVect(self.opts)
        self.X = vectors.sentence_representation(data)
        return self.X

    def serialize_sentence_vector(self, X):
        """Serializes vector representation of sentences"""
        
        filepath = './serializations/X_{}.npz'.format(self.opts['id'])
        if (os.path.isfile(filepath) is True):
            raise ValueError('File Already exists')
            
        if isinstance(X, numpy.ndarray):
            with open('./serializations/X_{}.npz'.format(self.opts['id']), 'wb') as handle:
                pickle.dump(X, handle)
        else:
            raise ValueError('Sentence vector type not expected')

    
    def read_serialized_sentences_vector(self, path):
        """Loads Serialized sentence vectors"""
        
        with open('{}X_{}.npz'.format(path, self.opts['id']), 'rb') as handle:
            self.X = pickle.load(handle)
        return self.X

    @decorate.elapsedtime_log
    def train_som(self, X):
        """Trains the SOM using the provided algorithm"""

        H = int(self.som_size)
        W = int(self.som_size)
        N = X.shape[1]
       
        if self.opts['algorithm'] in ['MINISOMBATCH', 'MINISOMRANDOM']:
            som = MiniSom(H, W, N, sigma=1.0, random_seed=1)

            if self.opts['initialization']:
                som.random_weights_init(X)

            if self.opts['algorithm'] == 'MINISOMBATCH':
                som.train_batch(X, self.opts['niterations'])
            elif self.opts['algorithm'] == 'MINISOMRANDOM':
                som.train_random(X, self.opts['niterations'])

            with open('./serializations/codebook_{}.npy'.format(self.opts['id']), 'wb') as handle:
                pickle.dump(som.get_weights(), handle)
        else:
            raise ValueError('unexpected algorithm')

        return som
       

if __name__ == '__main__':
    if len(sys.argv) != 1:
        print('USAGE: python {}'.format(sys.argv[0]))
        sys.exit(1)

    opts = {'id': 9, 'algorithm': 'MINISOMBATCH', 'initialization': True, 'size': 64, 'paragraph_length': 1500, 'niterations': 1000, 'n_components': 700, 'use_hashing' : False, 'use_idf' : True, 'n_features' : 10000, 'minibatch' : False, 'verbose' : False, 'testdataset': 'EN-RG-65'}
    
    sentencesom = SentenceSom(opts)
    
    #sentencesom.som_size
    #sentences = sentencesom.get_sentences()
    #sentencesom.serialize_sentences_text('./serializations/sentences/')
    #sentences = sentencesom.read_serialized_sentences_text('./serializations/sentences/')
    #X = sentencesom.create_sentence_vector(sentences)
    #sentencesom.serialize_sentence_vector(X)
    #X = sentencesom.read_serialized_sentences_vector('./serializations/')
    #sentencesom.train_som(X)

    dataset_enrg65 = testdataset.TestDataset(opts)
    words = dataset_enrg65.fetch('distinct_words')
    evaluation_set = dataset_enrg65.fetch('data')
    snippets_by_word = dataset_enrg65.get_snippets_by_word()
    # print (snippets_by_word['car'])

    # create fingerprint instance passing opts and snippets_by_word dictionary
    fingerprints_enrg65 = fp.FingerPrint(opts)
    # fingerprints_enrg65.create_fingerprints(snippets_by_word, words)
    fingerprints_enrg65.evaluate(evaluation_set, 'cosine')


    # print (help(SentenceSom))
    # print (sentencesom.get_sentences().shape)
    # print (sentencesom.get_sentences(1000).shape)



