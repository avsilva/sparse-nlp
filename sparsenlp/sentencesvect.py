from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from time import time


class SentenceVect():
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
    
    def __init__(self, opts):
        """Initializes a Sentence Vector Representation.

        """

        self.opts = opts

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
