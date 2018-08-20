from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
import sparsenlp.modelresults as modelres
import utils.decorators as decorate
from time import time
import pandas as pd
import numpy
import pickle


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

        self.opts = opts
        self.X = None
        self.sentences = None

    @decorate.elapsedtime_log
    def create_sentence_vector(self):
        """Creates vector representation of sentences
        
        Returns
        -------
        sparse matrix
            text vector representation
        """

        results = modelres.ModelResults('./logs')
        result_id = self.check_same_sentence_vector(results.get_results())
        #print(result_id)
        if result_id is not False and result_id != self.opts['id']:
            print ('Using existing vector representation: id {}'.format(result_id))
            with open('{}X_{}.npz'.format(self.path, result_id), 'rb') as handle:
                self.X = pickle.load(handle)
        else:
            sentences = self._read_serialized_sentences_text()
            self.X = self.sentence_representation(sentences.cleaned_text)
            self._serialize_sentence_vector()

        return self.X

    def _read_serialized_sentences_text(self):
        """Returns pandas dataframe text sentences"""
        
        try:
            self.sentences = pd.read_pickle('{}{}.bz2'.format(
                                            '{}sentences/'.format(self.path), 
                                            self.opts['paragraph_length']), 
                                            compression="bz2")
        except OSError as e:
            raise OSError('Sentences dataframe does not exists')
            
        return self.sentences

    def _serialize_sentence_vector(self):
        """Serializes vector representation of sentences"""
            
        if isinstance(self.X, numpy.ndarray):
            with open('./serializations/X_{}.npz'.format(self.opts['id']), 
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

    def check_same_sentence_vector(self, results):
        
        keys = ['paragraph_length', 'n_features', 'n_components', 'use_idf', 'use_hashing']

        for result in results:
            
            equal = True
            for key in keys:
                if result[key] != self.opts[key]:
                    equal = False
                    continue

            if equal is True:
                return result['id']
            
        return False
