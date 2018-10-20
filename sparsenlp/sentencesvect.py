from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
import sparsenlp.modelresults as modelres
import utils.decorators as decorate
from sparsenlp.datacleaner import DataCleaner
from time import time
import pandas as pd
import numpy as np
import numpy
import pickle
import sys
import os


class SentenceVect():
    
    path = './serializations/'

    """Sentence Vector Representation

        The instance is intended for further processing including vectorization and clustering

        Attributes
        ----------
        opts : dict
            instance settings (e.g n_features, use_idf)
        
        Methods
        -------
        sentence_representation()
            returns text vector representation

    """

    @decorate.create_log
    def __init__(self, opts):
        """Initializes a Sentence Vector Representation.

        """

        if 'sentecefolder' in opts:
            self.path = opts['sentecefolder']

        self.opts = opts
        self.X = None
        self.snippets_by_word = None
        self.sentences = None

   
    def get_word_snippets(self, word_snippets):

        print('Getting snippets by word: {}'.format(word_snippets))
        with open('{}/{}'.format(self.path, word_snippets), 'rb') as handle:
            snippets_by_word = pickle.load(handle)
        return snippets_by_word

    @decorate.elapsedtime_log
    def create_word_snippets(self, reference_dataset, word_snippets=None):
        """Creates dictonary with counts and sentence index for each benchmark word.
            
        Returns
        -------
        dict
            benchmark dataset words
        """

        testdataset = list(reference_dataset.keys())[0]
        words = list(reference_dataset.values())[0]
        words = [w.lower() for w in words]

        logs = modelres.ModelResults('./logs')
  
        excpt = self.opts['id']
        if 'new_log' in self.opts and self.opts['new_log'] is False:
            excpt = None

        results = logs.get_results(exception=excpt)
        word_snippets_id = self._check_same_word_snippets(results, testdataset)

        if 'repeat' in self.opts and self.opts['repeat'] is True:
            word_snippets_id = False

        if word_snippets_id is not False:
            print('Using existing snippets by word: snippets_by_word_{}_{}.pkl'.format(word_snippets_id, testdataset))
            with open('{}snippets_by_word_{}_{}.pkl'.format(self.path, word_snippets_id, testdataset), 'rb') as handle:
                snippets_by_word = pickle.load(handle)
        else:
            print('Creating new snippets by word: id {} for {}'.format(self.opts['id'], testdataset))
            #sentences = self._read_serialized_sentences_text()

            datacleaner = DataCleaner()
            ext = '12'+str(self.opts['dataextension'].replace(',', ''))
            dataframe_path = '{}dataframe_{}_text_{}.pkl'.format(self.path, ext, self.opts['paragraph_length'])
            dataframe = pd.read_pickle(dataframe_path, compression='bz2') 
            
            #snippets_by_word = self._get_snippets_and_counts(sentences, words)
            snippets_by_word = datacleaner.get_dataset_counts_as_is(dataframe, words)
            with open('{}snippets_by_word_{}_{}.pkl'.format(self.path, self.opts['id'], testdataset), 'wb') as f:
                pickle.dump(snippets_by_word, f)

        return snippets_by_word

    @decorate.elapsedtime_log
    def create_vectors(self):
        """Creates vector representation of sentences
        
        Returns
        -------
        sparse matrix
            text vector representation
        """

        logs = modelres.ModelResults('./logs')

        excpt = self.opts['id']
        if 'new_log' in self.opts and self.opts['new_log'] is False:
            excpt = None

        results = logs.get_results(exception=excpt)
        same_vectors = self._check_same_sentence_vector(results)

        if 'repeat' in self.opts and self.opts['repeat'] is True:
            same_vectors = []
        
        if len(same_vectors) > 0:
            log_id = min(same_vectors)
            print ('Using existing vector representation: id {}'.format(log_id))
            with open('{}X_{}.npz'.format(self.path, log_id), 'rb') as handle:
                self.X = pickle.load(handle)
        else:
            print ('Creating new vector representation: id {}'.format(self.opts['id']))
            
            datacleaner = DataCleaner()
            dataframe = self.get_dataframe()

            # from tokens list to text
            dataframe[self.opts['tokens']] = dataframe[self.opts['tokens']].apply(' '.join)
            self.X = self.sentence_representation(dataframe[self.opts['tokens']])
            self._serialize_sentence_vector()

        return self.X

    def get_dataframe(self):
        ext = '12'+str(self.opts['dataextension'].replace(',', ''))
        dataframe_path = '{}dataframe_{}_{}_{}_ok.pkl'.format(self.path, ext, self.opts['tokens'], self.opts['paragraph_length'])
        datacleaner = DataCleaner()
        if os.path.isfile(dataframe_path):
            print ('Reading dataframe {}'.format(dataframe_path))
            dataframe = pd.read_pickle(dataframe_path, compression='bz2') 
            print(dataframe.shape)
            print(dataframe.columns)
        else:
            print ('Creating new dataframe in {}'.format(dataframe_path))
            #self.opts['dataextension'] = ''
            #self.opts['paragraph_length'] = 500
            sentences = self._read_serialized_sentences_text()
            dataframe = sentences[['text']]     
            
            print('final sentences shape {}'.format(sentences.shape))
            #dataframe = datacleaner.tokenize_text(sentences)

            print('lemma or stemme dataframe')
            if self.opts['tokens'] == 'lemmas':
                dataframe = datacleaner.lemmatize_text(dataframe)
            elif self.opts['tokens'] == 'stemme':
                dataframe = datacleaner.steemer_text(dataframe)

            self._serialize_dataframe(dataframe, dataframe_path)

        return dataframe


    def _get_snippets_and_counts(self, _dataframe, _word):
        
        snippets_and_counts = {}
        for w in _word:
            info = {'idx': 0, 'counts': 0}
            snippets_and_counts[w] = [info]

        try:
            for index, row in _dataframe.iterrows():
                tokens = row[self.opts['tokens']].split()
                for w in _word:
                    
                    if tokens.count(w) != 0:
                        info = {'idx': index, 'counts': tokens.count(w)}
                        snippets_and_counts[w].append(info)
                
                if int(index) % 100000 == 0:
                    print('index {}'.format(index))

        except AttributeError as e:
            print (e)
        
        return snippets_and_counts

    def _read_serialized_sentences_text(self):
        """Returns pandas dataframe text sentences"""
        
        try:
            #self.sentences = pd.read_pickle('{}{}.bz2'.format('{}sentences/'.format(self.path), self.opts['paragraph_length']), 
            path = '{}sentences/{}_ok.bz2'.format(self.path, self.opts['paragraph_length'])
            self.sentences = pd.read_pickle(path, compression="bz2")
            #print (self.sentences.shape)
            if 'dataextension' in self.opts and self.opts['dataextension'] != '':
                extension_sentences = self._read_extension_sentences(self.opts['dataextension'])
                self.sentences = self.sentences.append(extension_sentences, ignore_index=True)
                del extension_sentences
            

        except OSError as e:
            raise OSError('Sentences dataframe does not exists')
            
        return self.sentences

    def _read_extension_sentences(self, dataextension):
        
        # create empty dataframe 
        new_sentences_df = pd.DataFrame(columns=self.sentences.columns)

        extensions = dataextension.split(',')
        for ext in extensions:
            folder = '{}sentences/articles{}/'.format(self.path, ext)
            file_list = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and f == 'articles{}_{}_ok.bz2'.format(ext, self.opts['paragraph_length'])]
            
            for file in file_list:
                df_sentences = pd.read_pickle('{}sentences/articles{}/{}'.format(
                                                self.path, ext, file), compression="bz2")
                new_sentences_df = new_sentences_df.append(df_sentences)
                del df_sentences
        return new_sentences_df
    
    def _serialize_dataframe(self, dataframe, dataframe_path):
        dataframe.to_pickle(dataframe_path, compression='bz2')
    
    def _serialize_sentence_vector(self):
        """Serializes vector representation of sentences"""
            
        if isinstance(self.X, numpy.ndarray):
            with open('{}X_{}.npz'.format(self.path, self.opts['id']), 
                      'wb') as handle:
                pickle.dump(self.X, handle, protocol=4)
        else:
            raise ValueError('Sentence vector type not expected')

    def sentence_representation(self, data):
        """Do the sentence vector representation.
        
        Parameters
        ---------
        data : numpy ndarray
            pandas series with text data (snippets)

        Returns
        -------
        numpy ndarray
            text vector representation
        """

        # TODO: refactor method: should be splited at least in two other methods
        if 'use_glove' in self.opts and self.opts['use_glove'] != False:
            
            sentences = []
            for tokens in data:
                #print (tokens.split(' '))
                sentences.append(tokens.split(' '))

            print ('Vectorizing sentences using GLOVE')
            glove = {w: x for w, x in self.gimme_glove()}
            X = [self.tokens_to_glove_vec(tokens, glove).mean(axis=0) for tokens in sentences]
            X = np.array(X)
            print (X.shape)

        else:
            print ('Vectorizing sentences using tf-idf or hashing')
            if self.opts['use_hashing']:
                if self.opts['use_idf']:
                    # Perform an IDF normalization on the output of HashingVectorizer
                    hasher = HashingVectorizer(n_features=self.opts['n_features'],
                                            stop_words='english', alternate_sign=False,
                                            norm=None, binary=False)
                    vectorizer = make_pipeline(hasher, TfidfTransformer())
                else:
                    vectorizer = HashingVectorizer(n_features=self.opts['n_features'],
                                                stop_words='english',
                                                alternate_sign=False, norm='l2',
                                                binary=False)
            else:
                vectorizer = TfidfVectorizer(max_df=0.5, max_features=self.opts['n_features'],
                                            min_df=2, stop_words='english',
                                            use_idf=self.opts['use_idf'])
                
            X = vectorizer.fit_transform(data)

            if self.opts['n_components']:
                print("Performing dimensionality reduction using LSA")
                t0 = time()
                # Vectorizer results are normalized, which makes KMeans behave as
                # spherical k-means for better results. Since LSA/SVD results are
                # not normalized, we have to redo the normalization.
                svd = TruncatedSVD(self.opts['n_components'])
                normalizer = Normalizer(copy=False)
                lsa = make_pipeline(svd, normalizer)

                X = lsa.fit_transform(X)

                print("done in %fs" % (time() - t0))

                explained_variance = svd.explained_variance_ratio_.sum()
                print("Explained variance of the SVD step: {}%".format(
                    int(explained_variance * 100)))

        return X

    def tokens_to_glove_vec(self, tokens, glove):
        words = [w for w in np.unique(tokens) if w in glove]
        if len(words) == 0:
            print ('EMPTY')
            words = ['all']
        return np.array([glove[w] for w in words])
    
    def gimme_glove(self):
        with open("./embeddings/glove.6B/{}.txt".format(self.opts['use_glove']), encoding='utf-8') as glove_raw:
            for line in glove_raw.readlines():
                splitted = line.split(' ')
                yield splitted[0], np.array(splitted[1:], dtype=np.float)

    def _check_same_word_snippets(self, results, testdataset):
        #keys = ['paragraph_length', 'dataextension', 'testdataset']
        keys = ['paragraph_length', 'dataextension', 'tokens']
        same_word_snippets = []
        for result in results:
            equal = True
            for key in keys:
                if (key not in result) or (result[key] != self.opts[key]):
                    equal = False
                    continue
            if equal is True:
                same_word_snippets.append(result['id'])

        for snippets_id in same_word_snippets:
            file = self._check_if_file_exists(snippets_id, testdataset)
            if file is not False:
                return snippets_id
            
        return False
    
    def _check_if_file_exists(self, snippets_id, testdataset):
        file = '{}snippets_by_word_{}_{}.pkl'.format(self.path, snippets_id, testdataset)
        if os.path.isfile(file):
            return snippets_id
        else:
            return False

    def _check_same_sentence_vector(self, results):
        
        keys = ['paragraph_length', 'dataextension', 'tokens', 'n_features', 'n_components', 'use_idf', 'use_hashing', 'use_glove']
        same_vectors = []
        for result in results:
            
            equal = True
            for key in keys:
                if (key not in result) or (result[key] != self.opts[key]):
                    equal = False
                    continue

            if equal is True:
                same_vectors.append(result['id'])
            
        return same_vectors

    """
    def _check_same_snippets_by_word(self, results):
        
        keys = ['paragraph_length', 'dataextension']

        same_snippets_by_word = []
        for result in results:
            
            equal = True
            for key in keys:
                if (key not in result) or (result[key] != self.opts[key]):
    
                    equal = False
                    continue

            if equal is True:
                same_snippets_by_word.append(result['id'])
            
        return same_snippets_by_word
    """