from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
import sparsenlp.modelresults as modelres
import utils.decorators as decorate
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
        
    def get_word_snippets(self):
        logs = modelres.ModelResults('./logs')
        results = logs.get_results(exception=self.opts['id'])
        same_word_snippets = self._check_same_word_snippets(results)
        if len(same_word_snippets) == 0:
            raise ValueError('Cannot get word snippets !!!')
        else:
            log_id = min(same_word_snippets)
            print('Using existing snippets by word: id {}'.format(log_id))
            with open('{}snippets_by_word_{}.pkl'.format(self.path, log_id), 'rb') as handle:
                snippets_by_word = pickle.load(handle)
            return snippets_by_word
        

    @decorate.elapsedtime_log
    def create_word_snippets(self, reference_dataset):
        """Creates dictonary with counts and sentence index for each benchmark word.
            
        Returns
        -------
        dict
            benchmark dataset words
        """

        testdataset = list(reference_dataset.keys())[0]
        testdataset_words = list(reference_dataset.values())[0]
        testdataset_words = [w.lower() for w in testdataset_words]

        logs = modelres.ModelResults('./logs')
  
        excpt = self.opts['id']
        if 'new_log' in self.opts and self.opts['new_log'] is False:
            excpt = None

        results = logs.get_results(exception=excpt)
        word_snippets_id = self._check_same_word_snippets(results, testdataset)

        if 'repeat' in self.opts and self.opts['repeat'] is True:
            #same_word_snippets = []
            word_snippets_id = False

        if word_snippets_id is not False:
            #log_id = min(same_word_snippets)
            print('Using existing snippets by word: snippets_by_word_{}_{}.pkl'.format(word_snippets_id, testdataset))
            with open('{}snippets_by_word_{}_{}.pkl'.format(self.path, word_snippets_id, testdataset), 'rb') as handle:
                snippets_by_word = pickle.load(handle)
        else:
            print('Creating new snippets by word: id {} for {}'.format(self.opts['id'], testdataset))
            sentences = self._read_serialized_sentences_text()

            snippets_by_word = self._get_snippets_and_counts(sentences, testdataset_words)
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

            sentences = self._read_serialized_sentences_text()            
    
            if 'token' not in self.opts:
                self.opts['token'] = 'cleaned_text'

            self.X = self.sentence_representation(sentences[self.opts['token']])
            #self.X = self.sentence_representation(sentences.cleaned_text)        
            #rawdata = self._get_train_data(sentences)
            #self.X = self.sentence_representation(rawdata)
            
            self._serialize_sentence_vector()

        return self.X

    def _get_train_data(self, dataframe):

        if self.opts['token'] == 'cleaned_text_all':
            dataframe['train'] = dataframe['cleaned_text'].str.replace('_',' ')
            dataframe['train'] = dataframe['cleaned_text'].str.lower()
        else:
            dataframe['train'] = dataframe[self.opts['token']]

        return dataframe['train']

    def _get_snippets_and_counts(self, _dataframe, _word):
        
        snippets_and_counts = {}
        for w in _word:
            info = {'idx': 0, 'counts': 0}
            snippets_and_counts[w] = [info]

        for index, row in _dataframe.iterrows():
            tokens = row[self.opts['tokens']].split()
            for w in _word:
                
                if tokens.count(w) != 0:
                    info = {'idx': index, 'counts': tokens.count(w)}
                    snippets_and_counts[w].append(info)

            if int(index) % 100000 == 0:
                print('index {}'.format(index))
        
        return snippets_and_counts

    def _read_serialized_sentences_text(self):
        """Returns pandas dataframe text sentences"""
        
        try:
            self.sentences = pd.read_pickle('{}{}.bz2'.format(
                                            '{}sentences/'.format(self.path), 
                                            self.opts['paragraph_length']), 
                                            compression="bz2")
            
            print(self.sentences.shape)
            if 'dataextension' in self.opts and self.opts['dataextension'] != '':
                extension_sentences = self._read_extension_sentences(self.opts['dataextension'])
                self.sentences = self.sentences.append(extension_sentences, ignore_index=False)
            print(self.sentences.shape)

        except OSError as e:
            raise OSError('Sentences dataframe does not exists')
            
        return self.sentences

    def _read_extension_sentences(self, dataextension):
        
        # create empty dataframe 
        new_sentences_df = pd.DataFrame(columns=self.sentences.columns)

        extensions = dataextension.split(',')
        for ext in extensions:
            folder = '{}sentences/articles{}/'.format(self.path, ext)
            file_list = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and f == 'articles{}_{}.bz2'.format(ext, self.opts['paragraph_length'])]
            print (file_list)
            for file in file_list:
                df_sentences = pd.read_pickle('{}sentences/articles{}/{}'.format(
                                                self.path, ext, file), compression="bz2")
                new_sentences_df = new_sentences_df.append(df_sentences)
        #new_sentences_df.query('text_length_tokenized > {}'.format(self.opts['paragraph_length']), inplace=True)
        #new_sentences_df = new_sentences_df[['id', 'tokenized']]
        #new_sentences_df = new_sentences_df.rename(index=str, columns={"tokenized": "cleaned_text"})
        return new_sentences_df
    
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