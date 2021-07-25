from io import FileIO
import os
import sys
import errno
import re
import string
import json
import datetime
import pandas as pd
import numpy as np
import pickle
import concurrent.futures
from functools import partial
import multiprocessing
import tqdm
import collections
import re
import time
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

#from utils.corpora import clean_text as tokenizer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize.regexp import regexp_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *

#nltk.download('stopwords')
tokenizer = RegexpTokenizer(r'\w+')
STOP = {'english': stopwords.words('english'), 'portuguese': stopwords.words('portuguese')}

class AbstractReader:
    def debug(self, msg):
        self.log("DEBUG", msg)

    def info(self, msg):
        self.log("INFO", msg)

    def error(self, msg):
        self.log("ERROR", msg)

    def log(self, level, msg):
        raise NotImplementedError()

    @staticmethod
    def _include_file(file: str, excludes: list):
        for needle in excludes:
            if file.find(needle) != -1:
                return False
            return True

    def read_files_in_folder(self, path: str, excludes: list):
        for subdir, dirs, files in os.walk(path):
            for file in files:
                try:
                    if self._include_file(file, excludes):
                        with open('{}/{}'.format(subdir, file), 'r', encoding='utf-8') as handle:
                            yield handle.readlines()
                except json.decoder.JSONDecodeError as e:
                    print(e)
                    self.errors.append({"file": file, "err": e})
                except UnicodeDecodeError as e:
                    self.errors.append({"file": file, "err": e})

class JsonReader(AbstractReader):
    def __init__(self):
        self.errors = []

    def read(self, level, msg):
        self.file.write("{}: {}".format(level, msg))

    def parse_line(self, x):
        return json.loads(x)

class RawTextReader(AbstractReader):
    def __init__(self):
        self.errors = []

    def read(self, level, msg):
        self.file.write("{}: {}".format(level, msg))

    def parse_line(self, x):
        
        try:
            date = x.split(',')[0]
            hour = x.split('-')[0].split(',')[1]
            user = x.split('-')[1].split(':')[0]
            text = x.split(':').pop()
        except IndexError as e:
            date = None
            hour = None
            user = None
            text = x
        return {'date': date, 'hour': hour, 'user': user, 'text': text}

class DataCleaner():
    """Initializes an instance of data cleaner object. Tokenizes text input data.

    Methods
    -------
    ingestfiles()
        Reads some kind of data source and imports into same kind of data structure
    """

    def __init__(self, folder):
        self.folder = folder
        self.errors = []
        self.data = None

    def statistics(self):
        shape = self.data.shape
        return {"shape": shape}

    def save(self, filename, mode='pickle'):
        if mode == 'pickle':
            self.data.to_pickle('{}/{}'.format(self.folder, filename))
            #with open('{}/{}'.format(self.folder, filename), 'wb') as f:
            #    pickle.dump(self.data, f)
        return self

    def open(self, filename):
        #with open('{}/{}'.format(self.folder, filename), 'rb') as f:
        #    self.data = pickle.load(f)
        pd.read_pickle('{}/{}'.format(self.folder, filename))
        return self

    def ingest_files_into(self, mode: str, format: str):
        """Reads some kind of data source and imports into same kind of data structure

        Parameters
        ----------
        mode : str
            ingest mode
        format: str
            file format

        Returns
        -----------
            data imported in the defined mode
        """
        if format == 'json':
            reader = JsonReader()
        elif format == 'rawtext':
            reader = RawTextReader()
        else:
            raise ValueError('Unknown input format')

        excludes = ['xml']
        data = []
        if os.path.isdir(self.folder):
            for index, lines in enumerate(reader.read_files_in_folder(self.folder, excludes)):
                data += [reader.parse_line(x) for x in lines]  
            
            if mode == 'dict':
                self.data = data
            elif mode == 'pandasdataframe':
                self.data = pd.DataFrame(data)
                self.data['text_length'] = self.data['text'].str.len()
            
        else:
            raise IOError(errno.ENOENT, 'Not a Folder', input)

        return self

    def autofill_dataframe(self, axis=0):
        self.data = self.data.ffill(axis = 0) 
        return self

    def clean2(self, text):
        
        tokens = text.lower()
        tokens = re.sub(r'\d+', '', tokens)
        tokens = ' '.join(word_tokenize(tokens))
        tokens = tokenizer.tokenize(tokens)
        tokens = [x for x in tokens if len(x) > 2]
        tokens = [x for x in tokens if x not in stopwords.words('english')]
        return tokens

    def lemmatize(self, text):

        lemmatizer = WordNetLemmatizer()
        lemas = [lemmatizer.lemmatize(token) for token in text]
        return lemas

    def steemer(self, text):
    
        stemmer = PorterStemmer()
        stems = [stemmer.stem(token) for token in text]
        return stems


    @staticmethod
    def _cleantext(text):
        punc_list = '!"#$%&()*+,-./:;<=>?@[\]^_{|}~' + '0123456789'
        t = str.maketrans(dict.fromkeys(punc_list, " "))
        text = text.replace("\n", " ").replace("\r", " ")
        text = text.translate(t)
        return text

    def clean_text(self):
        print("cleaning text")
        t1 = time.time()
        
        num_processes = 8
        with concurrent.futures.ProcessPoolExecutor(num_processes) as pool:
            self.data['text'] = list(tqdm.tqdm(pool.map(self._cleantext, self.data['text'], chunksize=1000), total=self.data.shape[0]))
        
        """
        punc_list = '!"#$%&()*+,-./:;<=>?@[\]^_{|}~' + '0123456789'
        t = str.maketrans(dict.fromkeys(punc_list, " "))
        self.data = self.data.assign(text = lambda x: (x['text'].replace("\n", " ").replace("\r", " ")))
        self.data = self.data.assign(text = lambda x: (x['text'].str.translate(t)))
        """
        t2 = time.time()
        print ("cleaning ", self.data.shape[0], ' rows, time = %.3f' %(t2-t1))
        return self

    @staticmethod
    def _stopwords(stopwords, text):
        pat = r'\b(?:{})\b'.format('|'.join(stopwords))
        return re.sub(pat, '', text)
    
    def remove_stopwords(self, lang, aditional=[]):
        print("remove stopwords")
        
        stopwords = STOP[lang]
        stopwords += aditional
        print(type(stopwords))

        t1 = time.time()
        
        num_processes = 8
        func = partial(self._stopwords, stopwords)
        with concurrent.futures.ProcessPoolExecutor(num_processes) as pool:
            self.data['text'] = list(tqdm.tqdm(pool.map(func, self.data['text'], chunksize=1000), total=self.data.shape[0]))
        
        t2 = time.time()
        print ("stopwords ", self.data.shape[0], ' rows, time = %.3f' %(t2-t1))
        return self

    @staticmethod
    def _remove_emoji(text):
        emoji = r"(?:[^\s])(?<![\w{ascii_printable}])".format(ascii_printable=string.printable)
        return re.sub(emoji, '', text)
    
    def remove_emoji(self):
        print("remove emoji")
        
        t1 = time.time()
        num_processes = 8        
        with concurrent.futures.ProcessPoolExecutor(num_processes) as pool:
            self.data['text'] = list(tqdm.tqdm(pool.map(self._remove_emoji, self.data['text'], chunksize=1000), total=self.data.shape[0]))
        
        t2 = time.time()
        print ("emoji ", self.data.shape[0], ' rows, time = %.3f' %(t2-t1))
        return self

    @staticmethod
    def _remove_words_by_length(limit, text):
        tokens = text.split(' ')
        tokens = [x for x in tokens if len(x) > limit]
        return ' '.join(tokens)
    
    def remove_words_by_length(self, limit):
        print("remove words by length")
        t1 = time.time()
        num_processes = 8
        func = partial(self._remove_words_by_length, limit)
        with concurrent.futures.ProcessPoolExecutor(num_processes) as pool:
            self.data['text'] = list(tqdm.tqdm(pool.map(func, self.data['text'], chunksize=1000), total=self.data.shape[0]))
        t2 = time.time()
        print ("remove words by length ", self.data.shape[0], ' rows, time = %.3f' %(t2-t1))
        return self

    @staticmethod
    def _tolower(text):
        return text.lower()

    def all_lower(self):   
        print("to lower")
        t1 = time.time()
        num_processes = 8
        with concurrent.futures.ProcessPoolExecutor(num_processes) as pool:
            self.data['text'] = list(tqdm.tqdm(pool.map(self._tolower, self.data['text'], chunksize=1000), total=self.data.shape[0]))
        t2 = time.time()
        print ("to lower ", self.data.shape[0], ' rows, time = %.3f' %(t2-t1))
        return self  
        
    @staticmethod
    def _mytokenizer(text):
        WORD = re.compile(r'\w+')
        tokens = WORD.findall(text)
        return tokens

    def tokenize_text(self):
        print("tokenizing text")
        t1 = time.time()
        #self.data["tokens"] = self.data.apply(lambda x: self._mytokenizer(x['text']), axis=1)
        num_processes = 6
        with concurrent.futures.ProcessPoolExecutor(num_processes) as pool:
            self.data['tokens'] = list(tqdm.tqdm(pool.map(self._mytokenizer, self.data['text'], chunksize=1000), total=self.data.shape[0]))
        t2 = time.time()
        print ("tokenize ", self.data.shape[0], ' rows, time = %.3f' %(t2-t1))
        return self

    def lemmatize_text(self, dataframe):

        num_processes = 6
        with concurrent.futures.ProcessPoolExecutor(num_processes) as pool:
            dataframe['lemmas'] = list(tqdm.tqdm(pool.map(self.lemmatize, dataframe['text'], chunksize=10), total=dataframe.shape[0]))
        #dataframe["lemmas"] = dataframe.loc[:,('text')].apply(self.lemmatize)
        return dataframe[["lemmas"]]

    def steemer_text(self, dataframe):
    
        num_processes = 6
        with concurrent.futures.ProcessPoolExecutor(num_processes) as pool:
            dataframe['stemme'] = list(tqdm.tqdm(pool.map(self.steemer, dataframe['text'], chunksize=10), total=dataframe.shape[0]))
        #dataframe["stemme"] = dataframe.loc[:,('text')].apply(self.steemer)
        return dataframe[["stemme"]]

    def get_dataset_counts_as_is(self, dataframe, words):
        snippets_and_counts = {}

        for w in words:
            info = {'idx': 0, 'counts': 0}
            snippets_and_counts[w] = [info]

        for index, row in dataframe.iterrows():
            
            #tokens = row['text']
            #match exactly each word. car will not match cars
            tokens = row['text'].split()        
            for w in words:
                cnt = tokens.count(w)
                if cnt != 0:
                    info = {'idx': index, 'counts': cnt}
                    snippets_and_counts[w].append(info)

            if int(index) % 100000 == 0:
                print('index {}'.format(index))
        return snippets_and_counts

    def get_dataset_counts_stemme(self, dataframe, words):
        
        stemmer = PorterStemmer()
        snippets_and_counts = {}
        for w in words:
            info = {'idx': 0, 'counts': 0}
            snippets_and_counts[w] = [info]

        for index, row in dataframe.iterrows():
            #tokens = row['text']
            tokens = row['text'].split()        
            token_stemmes = [stemmer.stem(token) for token in tokens]
            for w in words:

                if tokens.count(w) != 0:
                    stemm = stemmer.stem(w)
                    count_stemmes_in_doc = token_stemmes.count(stemm)                
                    info = {'idx': index, 'counts': count_stemmes_in_doc}
                    snippets_and_counts[w].append(info)

            if int(index) % 100000 == 0:
                print('index {}'.format(index))
        return snippets_and_counts

    def get_counter_as_is(self, dataframe):
        counter = []
        for index, row in dataframe.iterrows():
            cnt = collections.Counter()
            cnt['idx'] = index
            #tokens = row['text']
            tokens = row['text'].split()
            for w in tokens:
                cnt[w] += 1
            counter.append(cnt)

            if int(index) % 50000 == 0 and index != 0:
                print('index {}'.format(index))
        return counter

    def get_counter_lemmas(self, dataframe):
        counter = []
        lemmatizer = WordNetLemmatizer()
        #stemmer = PorterStemmer()
        
        time1 = datetime.datetime.now()
        for index, row in dataframe.iterrows():
            cnt = collections.Counter()
            cnt['idx'] = index
            tokens = row['text']
            #print (tokens)
            #stems = [stemmer.stem(token) for token in tokens]
            lemas = [lemmatizer.lemmatize(token) for token in tokens]
            #print (stems)
            tokens = list(set(tokens))
            for w in tokens:
                lema = lemmatizer.lemmatize(w)
                count_lemmas_in_doc = lemas.count(lema)
                cnt[w] += count_lemmas_in_doc
            counter.append(cnt)

            if int(index) % 50000 == 0 and index != 0:
                print('index {}'.format(index))
                time2 = datetime.datetime.now()
                print('time elapsed: {}'.format(time2 - time1))
                time1 = datetime.datetime.now()
        
        return counter

    def get_word_snippets(self, words, counter):
        snippets_by_word = {}
        i = 0
        print (len(words))
        for w in words:
            #print (w)
            info = {'idx': 0, 'counts': 0}
            snippets_by_word[w] = [info]
            for cnt in counter:
                
                if w in cnt:
                    info = {'idx': cnt['idx'], 'counts': cnt[w]}
                    snippets_by_word[w].append(info)

            if int(i) % 50 == 0:
                print('index {}'.format(i))
            i += 1
        return (snippets_by_word)
    
    def get_word_snippets2(self, counter):
        snippets_by_word = {}
        i = 0

        for cnt in counter:
                
            for key, value in cnt.items():
                #print(key, value)
                info = {'idx': cnt['idx'], 'counts': value}
                try:
                    snippets_by_word[key].append(info)
                except KeyError as e:
                    snippets_by_word[key] = [info]
            if int(i) % 10000 == 0:
                print('index {}'.format(i))
            i += 1

        return snippets_by_word
        
        for w in words:
            #print (w)
            info = {'idx': 0, 'counts': 0}
            snippets_by_word[w] = [info]
            for cnt in counter:
                
                if w in cnt:
                    info = {'idx': cnt['idx'], 'counts': cnt[w]}
                    snippets_by_word[w].append(info)

            if int(i) % 50 == 0:
                print('index {}'.format(i))
            i += 1
        return (snippets_by_word)

    



    

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
        print(self.data.shape)
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

    def serialize(self, path, filename):
        """Tokenines text column into new DataFrame column"""

        self.data.to_pickle('{}/{}'.format(path, filename), 
                            compression='bz2')

    def sentence_segmentation(self, input, re_paragraph_splitter):
        """Takes input and splits into a list of paragraphs (snippets)"""
        snippets = re.split(re_paragraph_splitter, input)
        return snippets

    def explode_dataframe_into_snippets(self, column, re_paragraph_splitter):
        """Takes instance data dataframe and returns another dataframe with
            each text row splited into snippets
        """
                
        sLength = len(self.data[column])
        i = 0

        sentences = []
        for index, row in self.data.iterrows():
            
            if index % 2000 == 0:
                print('exploding {} th row of {}'.format(index, sLength))

            snippets = self.sentence_segmentation(row[column], re_paragraph_splitter)
            for snippet in snippets:
                #new_df = new_df.append({'id': row['id'], 'text': snippet, 'text_length': len(snippet)}, ignore_index=True)
                sentences.append({'id': row['id'], 'text': snippet, 'text_length': len(snippet)})
                i += 1
        
        self.data = pd.DataFrame(sentences)
        return self
        

