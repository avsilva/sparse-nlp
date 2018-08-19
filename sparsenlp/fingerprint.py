import os
import sys
import numpy as np
import pickle
from PIL import Image
import operator
from functools import reduce
import concurrent.futures
import pandas as pd
from sklearn.datasets.base import Bunch
from scipy.spatial import distance
import scipy
from minisom import MiniSom
import utils.decorators as decorate
import itertools
from functools import partial
#from multiprocessing import Pool, RawArray
import multiprocessing as mp
    
var_dict = {}


def init_worker(H, W, N, codebook):
    # Using a dictionary is not strictly necessary. You can also
    # use global variables.
    var_dict['H'] = H
    var_dict['W'] = W
    var_dict['N'] = N
    var_dict['codebook'] = codebook

def create_fp2(word_vectors):

    SOM = MiniSom(var_dict['H'], var_dict['W'], var_dict['N'], sigma=1.0, random_seed=1)
    SOM._weights = var_dict['codebook']
    #a = np.zeros((var_dict['H'], var_dict['W']), dtype=np.int)

    idx = word_vectors['idx']
    bmu = SOM.winner(word_vectors['vector'])
    return {word_vectors['counts']: bmu}
        

def create_fp(word_vectors):
    
    SOM = MiniSom(var_dict['H'], var_dict['W'], var_dict['N'], sigma=1.0, random_seed=1)
    SOM._weights = var_dict['codebook']
    a = np.zeros((var_dict['H'], var_dict['W']), dtype=np.int)

    for key, value in word_vectors.items():
        #print (key, type(value), len(value))
        for val in value:
            idx = val['idx']
            bmu = SOM.winner(val['vector'])
            a[bmu[0], bmu[1]] += val['counts']
            
    return {key: a}


class FingerPrint():
    
    path = './serializations/'

    def __init__(self, opts):
        """Initializes a FingerPrint instance.

        The instance is intended fingerprint creation and benchmark evaluation

        Attributes
        ----------
        opts : dict
            instance settings (e.g paragraph_length, size, algorithm)

        Methods
        -------
        serialize_sentences(sound=None)
        Prints the animals name and what sound it makes

        """
        self.opts = opts
        self.id = opts['id']
        self.name = opts['testdataset']
        self.sentece_length = opts['paragraph_length']
        self.algos = {'KMEANS': self._kmeans, 'MINISOMBATCH': self._minisom,
                      'MINISOMRANDOM': self._minisom}
    
    @decorate.elapsedtime_log
    def create_fingerprints(self, snippets_by_word, words):
        """Creates fingerprint for each word.
        
        Attributes
        ----------
        snippets_by_word : dict
            dict with counts and sentence index for each word
        words : list
            words for which fingerprint will be created
        """
        
        if isinstance(words, str):
            words = words.split(',')

        self.algos[self.opts['algorithm']](snippets_by_word, words)

    def _kmeans(self, snippets_by_word, words):
        """Creates fingerprints using kmeans codebook.
        
        Attributes
        ----------
        snippets_by_word : dict
            dict with counts and sentence index for each word
        words : list
            words for which fingerprint will be created
        """
        
        with open('./serializations/codebook_{}.npy'.format(self.opts['id']), 'rb') as handle:
            codebook = pickle.load(handle)
        # print (codebook)

        with open('./serializations/X_{}.npz'.format(self.opts['id']), 'rb') as handle:
            X = pickle.load(handle)
        # print (X.shape)

        word_fingerprint = {} 
        for word in words:
            a = np.zeros(self.opts['size'] * self.opts['size'], dtype=np.int)
            word_counts = snippets_by_word[word]
            # print (word)
            
            for info in word_counts[1:]:
                # print (info)
                idx = info['idx']
                a[codebook[idx]] += info['counts']
            #print (a)
            a = self._sparsify_fingerprint(a)
            #print (a)
            word_fingerprint[word] = a
        
        self._create_dict_fingerprint_image(word_fingerprint, 'fp_{}'.format(self.opts['id']))

    def _minisom(self, snippets_by_word, words):
        """Creates fingerprints using minisom codebook.
        
        Attributes
        ----------
        snippets_by_word : dict
            dict with counts and sentence index for each word
        words : list
            words for which fingerprint will be created
        """
        
        H = int(self.opts['size'])
        W = int(self.opts['size'])
        N = int(self.opts['n_components'])
        with open('./serializations/codebook_{}.npy'.format(self.opts['id']), 'rb') as handle:
            codebook = pickle.load(handle)
        SOM = MiniSom(H, W, N, sigma=1.0, random_seed=1)
        SOM._weights = codebook

        with open('./serializations/X_{}.npz'.format(self.opts['id']), 'rb') as handle:
            X = pickle.load(handle)

        num_processes =  mp.cpu_count() -1

        """
        word_vectors = []
        for word in words:
            a = []
            word_counts = snippets_by_word[word]
            
            for info in word_counts[1:]:
                idx = info['idx']
                a.append({'idx': idx, 'counts': info['counts'], 'vector': X[idx]})
            print (len(a))    
            with mp.Pool(processes=num_processes, initializer=init_worker, initargs=(H, W, N, codebook)) as pool:
                results = pool.map(create_fp2, a)
                #print('Results (pool):\n', results)
        """
          
        word_vectors = []
        for word in words:
            a = []
            word_counts = snippets_by_word[word]
            
            for info in word_counts[1:]:
                idx = info['idx']
                a.append({'idx': idx, 'counts': info['counts'], 'vector': X[idx]})
            word_vectors.append({word: a})        

        with mp.Pool(processes=num_processes, initializer=init_worker, initargs=(H, W, N, codebook)) as pool:
            results = pool.map(create_fp, word_vectors)
            #results = pool.starmap(create_fp, zip(words, word_vectors))
            #print('Results (pool):\n', results)
            for fingerprint in results:
                #print (fingerprint)
                for key, value in fingerprint.items():
                    a = self._sparsify_fingerprint(value)
                    self._create_fp_image(a, key+'_', 'fp_{}'.format(self.opts['id']))
        

       
        
        """
        for word in words:
            word_counts = snippets_by_word[word]
            print(word_counts)
            a = np.zeros((H, W), dtype=np.int)

            for info in word_counts[1:]:
                # print (info)
                idx = info['idx']
                bmu = SOM.winner(X[idx])
                a[bmu[0], bmu[1]] += info['counts']

            # np.savetxt('./images/fp_67/'+word+'.txt', a, fmt='%10.0f')
            a = self._sparsify_fingerprint(a)
            self._create_fp_image(a, word, 'fp_{}'.format(self.opts['id']))
        """
    def _sparsify_fingerprint(self, a):
        
        #hist = np.histogram(a, bins=10, range=None, normed=False, weights=None, density=None)
        hist = np.histogram(a, bins='scott', range=None, normed=False, weights=None, density=None)
        #print (hist[0])
        #print (hist[1])

        sparsify_percentage = 0.02
        if len(a.shape) == 2:
            nvalues = a.shape[0] * a.shape[1]
        elif len(a.shape) == 1:
            nvalues = a.shape[0]
        maxpixels = nvalues * sparsify_percentage
        #print ('maxpixels:', maxpixels)

        actual_value = 0
        pixels_on = 0
        for idx, val in enumerate(reversed(hist[0])):
            actual_value += val
            if actual_value > maxpixels:
                lower_limit_index = idx
                break
            else:
                pixels_on += val

        pixels_on_missing = round(maxpixels - pixels_on)
        rev = list(reversed(hist[1]))

        print('filling missing pixels...'+str(pixels_on_missing))

        if pixels_on_missing >= 0:
            lower = rev[lower_limit_index + 1]
            higher = rev[lower_limit_index + 0]
            a_copy = np.copy(a)
        
            counter = 0
            for x in np.nditer(a_copy, op_flags=['readwrite']):
                if counter <= pixels_on_missing:
                    if x >= lower and x < higher:
                        x[...] = 1
                        counter += 1
                    else:
                        x[...] = 0 
                else:
                    x[...] = 0
        
        lower_count = rev[lower_limit_index]
        #print ('lower_count: ', lower_count)

        sparsify = lambda t: 1 if t > lower_count else 0
        
        vfunc = np.vectorize(sparsify)
        b = vfunc(a)
        return b

    def _create_dict_fingerprint_image(self, fp_dict, image_dir):
        
        if not os.path.exists("./images/{}".format(image_dir)):
            os.makedirs("./images/{}".format(image_dir))

        with open('./images/{}/dict_{}.npy'.format(image_dir, self.opts['id']), 'wb') as handle:
            pickle.dump(fp_dict, handle)

    def _create_fp_image(self, a, word, image_dir):
        
        if not os.path.exists("./images/"+image_dir):
            os.makedirs("./images/"+image_dir)

        im = Image.fromarray(a.astype('uint8'))

        if os.path.exists("./images/"+image_dir+"/"+word+".bmp"):
            print('Removing '+'./images/'+image_dir+'/'+word+'.bmp')
            os.remove("./images/"+image_dir+"/"+word+".bmp")
        
        im.save("./images/"+image_dir+"/"+word+".png")
        Image.open("./images/"+image_dir+"/"+word+".png").convert('RGB').save("./images/"+image_dir+"/"+word+".bmp")
        
        im = Image.open("./images/"+image_dir+"/"+word+".bmp")
        # calculate lookup table
        lut = self._equalize(im.histogram())
        # map image through lookup table
        im = im.point(lut)
        im.save("./images/"+image_dir+"/"+word+".bmp")
        os.remove("./images/"+image_dir+"/"+word+".png")

    def _equalize(self, h):
        
        lut = []
        for b in range(0, len(h), 256):

            # step size
            step = reduce(operator.add, h[b:b+256]) / 255

            # create equalization lookup table
            n = 0
            for i in range(256):
                lut.append(n / step)
                n = n + h[i+b]
        return lut

    # TODO: evaluate only some percentage of evaluation dataset
    # TODO: compare pair of words
    @decorate.update_result_log
    def evaluate(self, evaluation_set, measure):
        """Evaluates benchmark word fingerprints using the specified measure.
        
        Attributes
        ----------
        evaluation_set : list
            benchmark data (i.e. word pairs with associated human scores)
        measure : str
            measure for vector pair evaluation
        """

        # Define distance measures mapping
        distance_measures = {
            "cosine": self._cosine, "euclidean": self._euclidean, 
            "similarbits": self._similarbits, "ssim": self._struc_sim
        }

        w1 = evaluation_set[0]
        w2 = evaluation_set[1]
        score = evaluation_set[2]
        
        df = pd.DataFrame({0: w1, 1: w2, 2: score})
        bunch = Bunch(X=df.values[:, 0:2].astype("object"),
                      y=df.values[:, 2:].astype(np.float))

        if (self.opts['algorithm'] == 'KMEANS'):
            with open('./images/fp_{}/dict_{}.npy'.format(self.opts['id'], 
                      self.opts['id']), 'rb') as handle:
                kmeans_fp = pickle.load(handle)
            
            A = [kmeans_fp[word] for word in bunch.X[:, 0]]
            B = [kmeans_fp[word] for word in bunch.X[:, 1]]
        else:

            if measure == 'ssim':
                mode = '2darray'
                A = [self._get_fingerprint_from_image(word, mode)
                     for word in bunch.X[:, 0]]
                B = [self._get_fingerprint_from_image(word, mode)
                     for word in bunch.X[:, 1]]
            else:
                mode = 'flatten'
                A = np.vstack(self._get_fingerprint_from_image(word, mode) 
                              for word in bunch.X[:, 0])
                B = np.vstack(self._get_fingerprint_from_image(word, mode) 
                              for word in bunch.X[:, 1])

        measure_fnct = distance_measures[measure]

        predicted_scores = measure_fnct(A, B)
        result = scipy.stats.spearmanr(predicted_scores, bunch.y).correlation
        return result

    def _get_fingerprint_from_image(self, word, _mode):
        
        filepath = './images/fp_{}/{}.bmp'.format(self.opts['id'], word)
        im = Image.open(filepath)
        r, g, b = im.split()
        pix = np.array(r)

        if _mode == '2darray':
            result = pix
        elif _mode == 'flatten':
            np.place(pix, pix > 1, [1])
            pix = pix.flatten()
            result = pix

        return result

    def _cosine(self, A, B):
        """
        Computes the distance, and not the similarity. 
        Must subtract the value from 1 to get the similarity.
        """

        return np.array([1 - distance.cosine(v1, v2) for v1, v2 in zip(A, B)])

    def _euclidean(self, A, B):
        """Computes the euclidean distance btw 2 vectors."""

        return np.array([1 / distance.euclidean(v1, v2) for v1, v2 in zip(A, B)])

    def _struc_sim(self, A, B):
        """Computes the structural similarity btw 2 vectors."""

        return np.array([ssim(v1, v2, win_size=33) for v1, v2 in zip(A, B)])
        
    def _similarbits(self, A, B):
        """Computes the similar bits btw 2 vectors."""

        C = A*B
        return np.array([v1.sum() for v1 in C])  # sum number of 1 bits

    def _fetch_ENRG65(self, mode):
        """Fetches ENRG65 benchmark dataset.
        
        Parameters
        ---------
        mode : str
            'distinct_words' for getting all distinct words or 
            'data' for getting all benchmark data
        """

        filepath = './datasets/similarity/EN-RG-65.txt'
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
        """Fetches benchmark dataset.
        
        Parameters
        ---------
        mode : str
            'distinct_words' for getting all distinct words or 
            'data' for getting all benchmark data
        """

        dictionary = ''

        # TODO: implement fecth for other benchmark datasets
        if self.name == 'EN-RG-65':
            data = self._fetch_ENRG65(mode)
        return data

    def get_snippets_by_word(self, words):
        """Returns dict with counts and sentence index for each benchmark word.
        
        Parameters
        ---------
        words : list
            benchmark dataset words
        """
        
        sentences = self._read_serialized_sentences_text(
            '{}sentences/'.format(self.path))

        filepath = './serializations/snippets_by_word_{}.pkl'.format(self.id)
        if (os.path.isfile(filepath) is False):
            snippets_by_word = self._get_snippets_and_counts(sentences, words)
            with open('./serializations/snippets_by_word_{}.pkl'.format(self.id), 'wb') as f:
                pickle.dump(snippets_by_word, f)
        else:
            with open('./serializations/snippets_by_word_{}.pkl'.format(
                        self.id), 'rb') as handle:
                snippets_by_word = pickle.load(handle)

        return snippets_by_word

    def _read_serialized_sentences_text(self, path):
        """Loads Serialized dataframe text sentences"""
        
        return pd.read_pickle('{}{}.bz2'.format(path, self.sentece_length),
                              compression="bz2")

    def _get_snippets_and_counts(self, _dataframe, _word):
        
        snippets_and_counts = {}
        for w in _word:
            info = {'idx': 0, 'counts': 0}
            snippets_and_counts[w] = [info]

        for index, row in _dataframe.iterrows():
            tokens = row['cleaned_text'].split()
            for w in _word:
                
                if tokens.count(w) != 0:
                    info = {'idx': index, 'counts': tokens.count(w)}
                    snippets_and_counts[w].append(info)

            if index % 100000 == 0:
                print('index '+str(index))
        
        return snippets_and_counts



                
                
