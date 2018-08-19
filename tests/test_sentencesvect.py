import unittest
import pandas
import numpy
import sparsenlp.sentencesvect as sentencevect


class TestSentencesVect(unittest.TestCase):
    """Tests for `sentencesvect.py`."""

    def setUp(self):
        opts = {
                'id': 68, 
                'paragraph_length': 400, 
                'n_features': 10000, 'n_components': 700, 'use_idf': True, 'use_hashing': False, 
                'algorithm': 'MINISOMBATCH', 'initialization': False, 'size': 64, 'niterations': 1000, 'minibatch': True, 
                'verbose': False, 
                'testdataset': 'EN-RG-65'
        }
        self.vector = sentencevect.SentenceVect(opts)

    def test_check_same_sentence_vector(self):
        """Tests if setence vector already exists"""
        
        results = [{
                'id': 67, 
                'paragraph_length': 400, 
                'n_features': 10000, 'n_components': 700, 'use_idf': True, 'use_hashing': False, 
                'algorithm': 'MINISOMBATCH', 'initialization': False, 'size': 64, 'niterations': 1000, 'minibatch': True, 
                'verbose': False, 
                'testdataset': 'EN-RG-65'
        }]
        id = self.vector.check_same_sentence_vector(results)
        self.assertEqual(id, 67)

if __name__ == "__main__":
    unittest.main()
