import os
import sys
import errno
import re
import json
import pandas as pd
import numpy as np
import concurrent.futures
import multiprocessing
import tqdm
from utils.corpora import clean_text as tokenizer


class DataCleaner():
    """Initializes an instance of data cleaner object. Tokenizes text input data.


    Methods
    -------
    ingestfiles()
        Reads some kind of data source and imports into same kind of data structure
    """

    def __init__(self):
        self.data = None

    def _read_files_in_folder(self, path):
        data = []
        for subdir, dirs, files in os.walk(path):
            for file in files:
                with open('{}/{}'.format(subdir, file), 'r', encoding='utf-8') as handle:
                    datafile=handle.readlines()
                    data += [json.loads(x) for x in datafile]
                    
        return data

    def ingestfiles(self, input, output):
        """Reads some kind of data source and imports into same kind of data structure

        Parameters
        ----------
        path : str
            input data source
        out : str
            type of data structure returned 

        Returns
        -----------
            data imported in the defined data structure
        """
        if os.path.isdir(input):
            data = self._read_files_in_folder(input)
            if output == 'dict':
                result = data
            elif output == 'pandas':
                result = pd.DataFrame(data)
                result['text_length'] = result['text'].str.len()
        else:
            raise IOError(errno.ENOENT, 'Not a Folder', input)

        self.data = result
        return result

    def clean(self, text):
        cleaned_texts = tokenizer(text, remove_stop_words=True, 
                                  remove_punct=True, 
                                  lemmas=True, remove_numbers=True, 
                                  remove_spaces=True, 
                                  remove_2letters_words=True, 
                                  remove_apostrophe=True, method='spacy', 
                                  spacy_disabled_components=['tagger', 'parser'])
        return cleaned_texts

    def tokenize_pandas_column(self, column, num_processes=4):
        """Tokenines text column into new DataFrame column"""

        #num_processes = multiprocessing.cpu_count() - 1
        with concurrent.futures.ProcessPoolExecutor(num_processes) as pool:
            # self.data['tokenized'] = list(pool.map(self.clean, self.data['text'], chunksize=10))
            self.data['tokenized'] = list(tqdm.tqdm(pool.map(self.clean, self.data[column], chunksize=10), total=self.data.shape[0]))
            
        self.data['tokenized'] = self.data['tokenized'].apply(lambda x: ' '.join(x))
        self.data['text_length_tokenized'] = self.data['tokenized'].str.len()

        return self.data
        
        """
        for index, row in self.data.iterrows():
            
            if index % 2 == 0:
                print('tokenizing {} th row of {}'.format(index, sLength))
                    
            tokens = tokenizer(row[column], 
                               remove_stop_words=True, 
                               remove_punct=True, 
                               lemmas=True, 
                               remove_numbers=True, 
                               remove_spaces=True, 
                               remove_2letters_words=True, 
                               remove_apostrophe=True, 
                               method='spacy', 
                               spacy_disabled_components=['tagger', 'parser'])
  
            self.data.set_value(index, 'tokenized', ' '.join(tokens))
            self.data.set_value(index, 'text_length_tokenized', len(' '.join(tokens)))
            
        return self.data
        """

    def serialize(self, filename, path):
        """Tokenines text column into new DataFrame column"""

        self.data.to_pickle('{}/{}'.format(path, filename), 
                            compression='bz2')

    def sentence_segmentation(self, input, re_paragraph_splitter):
        """Takes input and splits into a list of paragraphs (snippets)"""

        snippets = re.split(re_paragraph_splitter, input)
        return snippets

    def explode_dataframe_in_snippets(self, column, re_paragraph_splitter):
        """Takes instance data dataframe and returns another dataframe with
            each text row splited into snippets
        
        """
                
        #new_df = pd.DataFrame(columns=self.data.columns)
        sLength = len(self.data[column])
        i = 0

        sentences = []
        for index, row in self.data.iterrows():
            
            if index % 50 == 0:
                print('exploding {} th row of {}'.format(index, sLength))

            snippets = self.sentence_segmentation(row[column], re_paragraph_splitter)
            for snippet in snippets:
                #new_df = new_df.append({'id': row['id'], 'text': snippet, 'text_length': len(snippet)}, ignore_index=True)
                sentences.append({'id': row['id'], 'text': snippet, 'text_length': len(snippet)})
                i += 1
        
        df = pd.DataFrame(sentences)
        self.data = df
        return df
        

