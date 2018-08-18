import unittest
import os
import pandas as pd
import numpy
import sparsenlp.datacleaner as datacleaner


class TestDataClean(unittest.TestCase):
    """Tests for `dataclean.py`."""

    def setUp(self):
        opts = {'id': 42, 'algorithm': 'KMEANS', 'initialization': True, 
                'size': 25, 'paragraph_length': 550, 'n_features': 10000, 
                'niterations': 1000, 'n_components': 700, 
                'use_hashing': False, 'use_idf': True, 
                'minibatch': False, 'verbose': False, 
                'testdataset': 'EN-RG-65'}
        self.datacleaner = datacleaner.DataCleaner()

    def test_ingestfiles_json_to_dict(self):
        """Reads recursively several json datafiles into python dict (integration test)"""

        input = 'F:/RESEARCH/TESE/corpora/wikifiles/01012018/json/enwiki-20180101-pages-articles3.xml-p88445p200507'
        output = 'dict'
        data = self.datacleaner.ingestfiles(input, output)
        self.assertIsInstance(data[0], dict)

    def test_ingestfiles_json_to_pandas(self):
        """Reads recursively several json datafiles into pandas DataFrame (integration test)"""

        input = 'F:/RESEARCH/TESE/corpora/wikifiles/01012018/json/enwiki-20180101-pages-articles3.xml-p88445p200507'
        output = 'pandas'
        self.datacleaner.ingestfiles(input, output)
        self.assertIsInstance(self.datacleaner.data, pd.DataFrame)

    def test_sentence_segmentation(self):
        """Tests if a sentence is correctly segmentized in paragraphs or snippets"""

        input = 'This is the first paragraph.\n\n\nThis is the second paragraph.'
        re_paragraph_splitter = '\n\n+'
        result = self.datacleaner.sentence_segmentation(input, re_paragraph_splitter)
        self.assertEqual(result, ['This is the first paragraph.', 'This is the second paragraph.'])

    def test_explode_dataframe_in_snippets(self):
        """Tests if a dataframe is correctly splited in text snippets"""

        df = pd.DataFrame({'id': [1], 
                            'title': ['My test'], 
                            'text': ['This is the first paragraph.\n\n\nThis is the second paragraph.']
                            })
        self.datacleaner.data = df
        column = 'text'
        re_paragraph_splitter = '\n\n+'
        result = self.datacleaner.explode_dataframe_in_snippets(column, re_paragraph_splitter)
        self.assertEqual(result['text'][0], 'This is the first paragraph.')
        self.assertEqual(result['text'][1], 'This is the second paragraph.')

    def test_tokenizes_pandas_text_column_into_new_column(self):
        """Tests if string in dataframe column is well tokeninzed into a new dataframe sting column"""
        
        df = pd.DataFrame({'id': [1], 'title': ['My test'], 'text_length': 10, 'text': ['This is a functional test. It Should return this sentece tokenized.']})
        self.datacleaner.data = df
        data = self.datacleaner.tokenize_pandas_column('text')
        self.assertEqual(data['tokenized'][0], 'This functional test Should return sentece tokenized')

    def test_serializes_pandas_dataframe_to_file(self):
        """Tests if dataframe is well serialized ? (integration test)"""

        df = pd.DataFrame({'id': [1], 
                            'title': ['My test'], 
                            'text': ['This is a functional test. It Should return this sentece tokenized.'],
                            'tokenized': ['This functional test Should return sentece tokenized']
                            })
        filename = 'testpandas.bz2'
        path = './tests/'
        self.datacleaner.data = df
        data = self.datacleaner.serialize(filename, path)
        self.assertTrue(os.path.isfile('{}{}'.format(path, filename)))
        df2 = pd.read_pickle('{}{}'.format(path, filename), compression="bz2")
        self.assertEqual(df['tokenized'][0], df2['tokenized'][0])

if __name__ == "__main__":
    unittest.main()
