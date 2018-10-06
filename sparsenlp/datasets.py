import os
import json
import ast

    


class Datasets():
    
    filepath = {'EN-RG-65': './datasets/similarity/EN-RG-65.txt', 
                'EN-WS353': './datasets/similarity/EN-WS353.txt',
                'EN-TRUK': './datasets/similarity/EN-TRUK.txt',
                'EN-SIM999': './datasets/similarity/EN-SIM999.txt',
                'EN-MEN-LEM': './datasets/similarity/EN-MEN-LEM.txt',
                'EN-WSR353R': './datasets/similarity/EN-WSR353R.txt',
                'EN-WSS353S': './datasets/similarity/EN-WSS353S.txt',
                'EN-RW': './datasets/similarity/EN-RW.txt',
                }

    def __init__(self, name, header, delimiter, score_index, remove_end_chars):
        """Initializes benchmark Datasets instance.

        """
        
        self.name = name
        self.header = header
        self.delimiter = delimiter
        self.score_index = score_index
        self.remove_end_chars = remove_end_chars
        self.path = self.filepath[self.name]

    def factory(type):
        #return eval(type + "()")
        if type == "EN-RG-65": 
            return RG65()
        #if type == "EN-WS353": 
        if type in ["EN-WS353", "EN-WSR353R", "EN-WSS353S"]: 
            return WS353(type)
        if type == "EN-TRUK":
            return TRUK()
        if type == "EN-SIM999":
            return SIM999()
        if type == "EN-MEN-LEM":
            return MEN()
        if type == "EN-RW":
            return RW()
        assert 0, "Bad dataset creation: " + type
         
    def get_data(self, mode):
        """Fetches benchmark dataset.
        
        Parameters
        ---------
        mode : str
            'distinct_words' for getting all distinct words or 
            'data' for getting all benchmark data
        """

        #if self.name == 'EN-RG-65':
        #    data = self._fetch_rg65(mode)
        #elif self.name == 'EN-WS353':
        #    data = self._fetch_ENWS353(mode)
        data = self._fetch(mode)
        return {self.name: data}
    
    def _fetch_ENWS353(self, mode):
        pass
    
    
    def read_file(self):

        score = []
        w1 = []
        w2 = []
        with open(self.path, 'r', encoding='utf-8') as file:
            #print (len(file.readlines()))
            
            if self.header is False:
                lines = file.readlines()
            else:
                lines = file.readlines()[1:]

            for line in lines:
                data = self._get_words_in_dataset(line)
                w1.append(data[0])
                w2.append(data[1])
                score.append(data[2])
        
        return w1, w2, score

    
    def _fetch(self, mode):
        """Fetches ENRG65 benchmark dataset.
        
        Parameters
        ---------
        mode : str
            'distinct_words' for getting all distinct words or 
            'data' for getting all benchmark data
        """
        w1, w2, score = self.read_file()
        
        if mode == 'distinct_words':
            words = w1 + w2
            dictionary = set(words)
            data = list(dictionary)
        elif mode == 'data':
            data = [w1, w2, score]

        return data

    def _get_words_in_dataset(self, line):
        words = line.split(self.delimiter)
        w1 = words[0]
        w2 = words[1]
        if self.remove_end_chars != 0:
            w1 = w1[0:self.remove_end_chars]
            w2 = w2[0:self.remove_end_chars]
        
        score = float(words[self.score_index].replace('\n', ''))
        #score = score * 10.0 / 4.0
        return [w1, w2, score]


class RG65(Datasets):
    
    def __init__(self):
        self.name = 'EN-RG-65'
        super().__init__(self.name, header=False, delimiter='\t', score_index=2, remove_end_chars=0)
        

class WS353(Datasets):
    
    def __init__(self, name):
        self.name = name
        super().__init__(self.name, header=True, delimiter='\t', score_index=2, remove_end_chars=0)

class TRUK(Datasets):
    
    def __init__(self):
        self.name = 'EN-TRUK'
        super().__init__(self.name, header=False, delimiter=' ', score_index=2, remove_end_chars=0)

class SIM999(Datasets):
    
    def __init__(self):
        self.name = 'EN-SIM999'
        super().__init__(self.name, header=True, delimiter='\t', score_index=3, remove_end_chars=0)

class MEN(Datasets):
    
    def __init__(self):
        self.name = 'EN-MEN-LEM'
        super().__init__(self.name, header=False, delimiter=' ', score_index=2, remove_end_chars=-2)

class RW(Datasets):
    
    def __init__(self):
        self.name = 'EN-RW'
        super().__init__(self.name, header=False, delimiter='\t', score_index=2, remove_end_chars=0)
       

