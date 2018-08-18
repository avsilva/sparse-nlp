import os
import sys
import datetime
from sparsenlp.sentencecluster import SentenceCluster
from sparsenlp.fingerprint import FingerPrint
from sparsenlp.datacleaner import DataCleaner


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('USAGE: python {} mode'.format(sys.argv[0]))
        sys.exit(1)

    mode = sys.argv[1]
    print(mode)
    
    if mode == 'tokenize':
        datacleaner = DataCleaner()
        folder = 'F:/RESEARCH/TESE/corpora/wikifiles/01012018/json/enwiki-20180101-pages-articles3.xml-p88445p200507/AE'
        print ('ingesting files')
        datacleaner.ingestfiles(folder, 'pandas')    
        datacleaner.data = datacleaner.data[:5]
        print ('exploding dataframe_in_snippets')
        datacleaner.explode_dataframe_in_snippets('text', '\n\n+')
        time1 = datetime.datetime.now()
        datacleaner.tokenize_pandas_column('text')
        datacleaner.serialize('articles3_AE.bz2', 'F:/RESEARCH/TESE/corpora/wikifiles/01012018/json/')
        time2 = datetime.datetime.now()
        print(time2 - time1)
        
    elif mode == 'cluster':
        
        opts = {'id': 67, 'algorithm': 'MINISOMBATCH', 'initialization': False, 
                'size': 64, 'paragraph_length': 400, 'n_features': 10000, 
                'niterations': 1000, 'n_components': 700, 
                'use_hashing': False, 'use_idf': True, 'minibatch': True, 
                'verbose': False, 'testdataset': 'EN-RG-65'}
    
        # codebook creation
        mycluster = SentenceCluster(opts)
        #print (help(mycluster))
        #mycluster.serialize_sentences()
        mycluster.set_sentence_vector()
        mycluster.cluster()
        
        benchmarkdata = FingerPrint(opts)
        # fingerprints creation
        words = benchmarkdata.fetch('distinct_words')
        snippets_by_word = benchmarkdata.get_snippets_by_word(words)
        benchmarkdata.create_fingerprints(snippets_by_word, words)

        # evaluation
        evaluation_data = benchmarkdata.fetch('data')
        benchmarkdata.evaluate(evaluation_data, 'cosine')

    else:
        raise ValueError('wrong mode !!!')