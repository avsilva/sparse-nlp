import os
import sys
import pickle
sys.path.append(os.path.abspath('./utils'))
import database as db
import corpora as corp

DIR = 'C:/Users/andre.silva/web_data/'


class TestDataset():
    
    def __init__(self, opts):
        """Initializes a test Dataset."""

        self.name = opts['testdataset']
        self.id = opts['id']
        self.sentece_length = opts['paragraph_length']

    def _fetch_ENRG65(self, mode):
        filepath = '{}/similarity/EN-RG-65.txt'.format(DIR)
        file = open(filepath, 'r', encoding='utf-8')
        score = []
        w1 = []
        w2 = []
        for line in file:
            data = self._get_words_for_rg65_dataset(line)
            w1.append(data[0])
            w2.append(data[1])
            score.append(data[2])

        if mode == 'distinct_words':
            words = w1 + w2
            dictionary = set(words)
            self.words = list(dictionary)
            return self.words
        elif mode == 'data':
            return [w1, w2, score]

    def _get_words_for_rg65_dataset(self, line):
        words = line.split('\t')
        w1 = words[0]
        w2 = words[1]
        score = float(words[2].replace('\n', ''))
        return [w1, w2, score]

    def fetch(self, mode):
        dictionary = ''
        if self.name == 'EN-RG-65':
            data = self._fetch_ENRG65(mode)
        return data

    def get_snippets_by_word(self):
        
        words = self.fetch('distinct_words')

        filepath = './serializations/snippets_by_word_{}.pkl'.format(self.id)
        if (os.path.isfile(filepath) is False):
            dataframe = db.get_cleaned_data(None, self.sentece_length)
            print('dataframe shape {} '.format(dataframe.shape))
            snippets_by_word = corp.get_snippets_and_counts(dataframe, words)

            with open('./serializations/snippets_by_word_{}.pkl'.format(self.id), 'wb') as f:
                pickle.dump(snippets_by_word, f)
        else:
            with open('./serializations/snippets_by_word_{}.pkl'.format(self.id), 'rb') as handle:
                snippets_by_word = pickle.load(handle)

        return snippets_by_word

    