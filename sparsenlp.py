import os
import sys
from sparsenlp.sentencecluster import SentenceCluster
from sparsenlp.fingerprint import FingerPrint
from sparsenlp.testdataset import TestDataset


if __name__ == '__main__':
    if len(sys.argv) != 1:
        print('USAGE: python {}'.format(sys.argv[0]))
        sys.exit(1)

    opts = {'id': 2, 'algorithm': 'KMEANS', 'initialization': True, 'size': 30, 'paragraph_length': 400, 'n_features': 10000, 'niterations': 1000, 'n_components': 700, 'use_hashing' : False, 'use_idf' : True, 'minibatch' : True, 'verbose' : False, 'testdataset': 'EN-RG-65'}
    
    mycluster = SentenceCluster(opts)
    #print (help(mycluster))
    
    #mycluster.serialize_sentences()
    #sys.exit(0)
    
    #mycluster.set_sentence_vector()
    #mycluster.cluster()
    #sys.exit(0)
    
    
    benchmarkdata = FingerPrint(opts)
    
    """
    words = benchmarkdata.fetch('distinct_words')
    snippets_by_word = benchmarkdata.get_snippets_by_word(words)
    benchmarkdata.create_fingerprints(snippets_by_word, words)
    """

    evaluation_data = benchmarkdata.fetch('data')
    benchmarkdata.evaluate(evaluation_data, 'cosine')