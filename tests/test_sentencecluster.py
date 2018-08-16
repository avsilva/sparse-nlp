import unittest
import pandas
import numpy
import sparsenlp.sentencecluster as sentencecluster


class TestSentenceCluster(unittest.TestCase):
    """Tests for `sentencecluster.py`."""

    def setUp(self):
        opts = {'id': 42, 'algorithm': 'KMEANS', 'initialization': True, 
                'size': 25, 'paragraph_length': 550, 'n_features': 10000, 
                'niterations': 1000, 'n_components': 700, 
                'use_hashing': False, 'use_idf': True, 
                'minibatch': False, 'verbose': False, 
                'testdataset': 'EN-RG-65'}
        self.cluster = sentencecluster.SentenceCluster(opts)

    def test_serialize_sentences(self):
        """A pandas dataframe is correctly returned?"""
        
        dataframe = self.cluster.serialize_sentences()
        self.assertIsInstance(dataframe, pandas.core.frame.DataFrame)
    
    def test_set_sentence_vector(self):
        """A sparse crs is correctly returned?"""
        
        X = self.cluster.set_sentence_vector()
        self.assertIsInstance(X, numpy.ndarray)

    def test_cluster_with_kmeans_minibatch(self):
        """N cluster's labels are correctly returned for each vector?"""

        self.cluster.opts['minibatch'] = True
        size = self.cluster.opts['size'] * self.cluster.opts['size']
        X = self.cluster.set_sentence_vector()
        cluster_labels = self.cluster.create_kmeans_minibatch_cluster(size)
        self.assertEqual(X.shape[0], len(cluster_labels))

if __name__ == "__main__":
    unittest.main()
