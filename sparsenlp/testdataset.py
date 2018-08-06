import os
import sys
import pickle
sys.path.append(os.path.abspath('./utils'))
import database as db
import corpora as corp

DIR = './datasets/'


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

    def get_snippets_by_word(self, sentences):
        
        words = self.fetch('distinct_words')

        filepath = './serializations/snippets_by_word_{}.pkl'.format(self.id)
        if (os.path.isfile(filepath) is False):
            snippets_by_word = self.get_snippets_and_counts(sentences, words)
            with open('./serializations/snippets_by_word_{}.pkl'.format(self.id), 'wb') as f:
                pickle.dump(snippets_by_word, f)
        else:
            with open('./serializations/snippets_by_word_{}.pkl'.format(self.id), 'rb') as handle:
                snippets_by_word = pickle.load(handle)

        return snippets_by_word

    def get_snippets_and_counts(self, _dataframe, _word):
        
        snippets_and_counts = {}
        for w in _word:
            info = {'idx': 0, 'counts': 0}
            snippets_and_counts[w] = [info]

        for index, row in _dataframe.iterrows():
            tokens = row['cleaned_text'].split()
            for w in _word:
                
                if tokens.count(w) != 0:
                    info = {'idx': index, 'counts': tokens.count(w)}
                    #if w not in snippets_and_counts:
                    #    snippets_and_counts[w] = [info]
                    #else:
                    snippets_and_counts[w].append(info)

            if index % 100000 == 0:
                print ('index '+str(index))
        
        return snippets_and_counts

    