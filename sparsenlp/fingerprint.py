import os
import sys
import numpy as np
import pickle
from PIL import Image
import operator
from functools import reduce
import concurrent.futures
import pandas as pd
import datetime
from sklearn.datasets.base import Bunch
try:
    from skimage.measure import compare_ssim
except:
    print('ERROR: ' +str(sys.exc_info()[0]))

from scipy.stats import wasserstein_distance
from scipy.spatial import distance
import scipy
from minisom import MiniSom
import utils.decorators as decorate
import sparsenlp.modelresults as modelres
from sparsenlp.sentencesvect import SentenceVect
from sparsenlp.calculations import *
from sparsenlp.sentencecluster import SentenceCluster
import itertools
# from functools import partial
import multiprocessing as mp
try:
    import dask.multiprocessing
    from dask import compute, delayed
except:
    print('ERROR: ' +str(sys.exc_info()[0]))
import random


class FingerPrint():
    
    path = './serializations/'

    def __init__(self, opts, mode='numba'):
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
        self.mode = mode
        self.algos = {'KMEANS': self._kmeans, 'MINISOMBATCH': self._minisom,
                      'MINISOMRANDOM': self._minisom}
    
    @decorate.elapsedtime_log
    def create_fingerprints(self, snippets_by_word=None, X=None, codebook=None, words=None, fraction=None):
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
        elif words is None:
            words = list(snippets_by_word.keys())

        words = self._check_existing_word_fp('fp_{}'.format(self.opts['id']), words, float(fraction))
        print ('Creating SDR for {} words'.format(len(words)))
        self.algos[self.opts['algorithm']](snippets_by_word, words, X, codebook)

        return True

    def _kmeans(self, snippets_by_word, words, X, codebook):
        """Creates fingerprints using kmeans codebook.
        
        Attributes
        ----------
        snippets_by_word : dict
            dict with counts and sentence index for each word
        words : list
            words for which fingerprint will be created
        """
        
        """
        with open('./serializations/codebook_{}.npy'.format(self.opts['id']), 'rb') as handle:
            codebook = pickle.load(handle)
        

        with open('./serializations/X_{}.npz'.format(self.opts['id']), 'rb') as handle:
            X = pickle.load(handle)
        """

        word_fingerprint = {} 
        for word in words:
            a = np.zeros(self.opts['size'] * self.opts['size'], dtype=np.int)
            word_counts = snippets_by_word[word]
            # print (word)
            
            for info in word_counts[1:]:
                # print (info)
                idx = info['idx']
                a[codebook[idx]] += info['counts']
            #print (word)
            a = self._sparsify_fingerprint(a)
            #print (a)
            word_fingerprint[word] = a
        
        self._create_dict_fingerprint_image(word_fingerprint, 'fp_{}'.format(self.opts['id']))

    def _get_unique_word_vectors(self, unique_indexes, X):
        unique_word_vectors = []
        for idx in unique_indexes:
            unique_word_vectors.append({'idx': idx, 'vector': X[idx]})
        return unique_word_vectors

    def _tranform_list_to_dict(self, results):
        idx_vectors = {} 
        for result in results:
            for key in result:
                idx_vectors[key] = result[key]
        return idx_vectors

    def _create_fp_for_words(self, snippets_by_word, words, idx_vectors, H, W):
        for word in words:
            a = np.zeros((H, W), dtype=np.int)
            word_counts = snippets_by_word[word]
            
            for info in word_counts[1:]:
                idx = info['idx']
                bmu = idx_vectors[idx]
                a[bmu[0], bmu[1]] += info['counts']
            print(word)
            a = self._sparsify_fingerprint(a)
            self._create_fp_image(a, word, 'fp_{}'.format(self.opts['id']))


    def _minisom(self, snippets_by_word, words, X, codebook):
        """Creates fingerprints using minisom codebook.
        
        Attributes
        ----------
        snippets_by_word : dict
            dict with counts and sentence index for each word
        words : list
            words for which fingerprint will be created
        """

        H = int(codebook.shape[0])
        W = int(codebook.shape[1])
        N = X.shape[1]

        SOM = MiniSom(H, W, N, sigma=1.0, random_seed=1)
        SOM._weights = codebook

        
        unique_indexes = set()
        word_vectors = []
        #print (words)
        #words = ['cock', 'rooster']
        
        for word in words:
            a = []
            word_counts = snippets_by_word[word]
            
            for info in word_counts[1:]:
                idx = info['idx']
                #print ('idx {}'.format(idx))
                a.append({'idx': idx, 'counts': info['counts'], 'vector': X[idx]})
                unique_indexes.add(idx)
            word_vectors.append({word: a})
        
        
        if self.mode == 'ckdtree':
            print ('using ckdtree')
            unique_word_vectors = self._get_unique_word_vectors(unique_indexes, X)
            codebook = np.reshape(codebook, (codebook.shape[0] * codebook.shape[1], codebook.shape[2]))
            values = [delayed(process_ckdtree)(codebook, x, H, W) for x in unique_word_vectors]
            results = compute(*values, scheduler='processes')
            
            idx_vectors = self._tranform_list_to_dict(results)
            self._create_fp_for_words(snippets_by_word, words, idx_vectors, H, W)

        elif self.mode == 'numba':
            print ('using numba')

            """
            values = [delayed(process3)(codebook, x, H, W) for x in word_vectors]
            results = compute(*values, scheduler='processes')

            for fingerprint in results:
                for key, value in fingerprint.items():
                    a = self._sparsify_fingerprint(value)
                    self._create_fp_image(a, key, 'fp_{}'.format(self.opts['id']))

            """
            unique_word_vectors = self._get_unique_word_vectors(unique_indexes, X)
            values = [delayed(process2)(codebook, x, H, W) for x in unique_word_vectors]
            results = compute(*values, scheduler='processes')
            
            idx_vectors = self._tranform_list_to_dict(results)
            a = np.zeros((H, W), dtype=np.int)
            self._create_fp_for_words(snippets_by_word, words, idx_vectors, H, W)
            

        elif self.mode == 'multiprocess':

            num_processes = mp.cpu_count() - 1
            with mp.Pool(processes=num_processes, initializer=init_worker, initargs=(H, W, N, codebook)) as pool:
                results = pool.map(create_fp, word_vectors)
                #print('Results (pool):\n', results)
                for fingerprint in results:
                    #print (fingerprint)
                    for key, value in fingerprint.items():
                        a = self._sparsify_fingerprint(value)
                        self._create_fp_image(a, key, 'fp_{}'.format(self.opts['id']))

        
            

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
    def _check_existing_word_fp(self, image_dir, words, fraction=None):
        
        if not os.path.exists("./images/"+image_dir):
            pass
        else:
            i = 0
            ind2remove = []
            for word in words:
                if os.path.exists("./images/"+image_dir+"/"+word+".bmp"):
                    ind2remove.append(i)
                i += 1 
            words = [x for i, x in enumerate(words) if i not in ind2remove]

        if fraction is not None:
            random.seed()
            s = random.randint(1, 1000000)
            n_samples = round(len(words) * fraction)
            words = random.sample(words, n_samples)
        
        return words
                    
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

        with open('./images/{}/dict_{}_{}.npy'.format(image_dir, self.opts['id'], self.opts['dataset']), 'wb') as handle:
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

        #print('Evaluating fingerperints using SDRs from images/fp_{}'.format(self.opts['id']))

        
        testdataset = list(evaluation_set.values())[0]
        datasetname = list(evaluation_set.keys())[0]

        # Define distance measures mapping
        distance_measures = {
            "cosine": self._cosine, 
            "euclidean": self._euclidean, 
            "similarbits": self._similarbits, 
            "structutal similarity": self._struc_similatity,
            'earth movers distance': self._wasserstein,
        }

        
        w1 = [w.lower() for w in testdataset[0]]
        w2 = [w.lower() for w in testdataset[1]]
        score = testdataset[2]

        """
        w1_ = [] 
        w2_ = [] 
        score_ = [] 
        for a, b, c in zip(w1, w2, score):
            if (os.path.isfile('./images/fp_{}/{}.bmp'.format(self.opts['id'], a)) is True) and (os.path.isfile('./images/fp_{}/{}.bmp'.format(self.opts['id'], b)) is True):
                w1_.append(a)
                w2_.append(b)
                score_.append(c)

        test_percentage =  round((len(w1_)/len(w1)) * 100, 1)
        print ('test percentage = {} %'.format(test_percentage))
        """
        df = pd.DataFrame({0: w1, 1: w2, 2: score})
        bunch = Bunch(X=df.values[:, 0:2].astype("object"),
                      y=df.values[:, 2:].astype(np.float))

        if (self.opts['algorithm'] == 'KMEANS'):

            A, B = self._get_kmeans_embeddings(bunch, datasetname)
            test_percentage = 100

        else:

            w1_ = [] 
            w2_ = [] 
            score_ = [] 
            for a, b, c in zip(w1, w2, score):
                if (os.path.isfile('./images/fp_{}/{}.bmp'.format(self.opts['id'], a)) is True) and (os.path.isfile('./images/fp_{}/{}.bmp'.format(self.opts['id'], b)) is True):
                    w1_.append(a)
                    w2_.append(b)
                    score_.append(c)

            test_percentage =  round((len(w1_)/len(w1)) * 100, 1)
            
            
            df = pd.DataFrame({0: w1_, 1: w2_, 2: score_})
            bunch = Bunch(X=df.values[:, 0:2].astype("object"),
                      y=df.values[:, 2:].astype(np.float))


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
        
        return {'score': result, 'percentage': test_percentage}

    def _get_kmeans_embeddings(self, bunch, datasetname):

        filepath = './images/fp_{}/dict_{}_{}.npy'.format(self.opts['id'], self.opts['id'], datasetname)
        if (os.path.isfile(filepath) is False):
            A = [0]
            B = [0]
        else:
            with open(filepath, 'rb') as handle:
                kmeans_fp = pickle.load(handle)

            A = [kmeans_fp[word] for word in bunch.X[:, 0]]
            B = [kmeans_fp[word] for word in bunch.X[:, 1]]

        return A, B

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

            if len(np.unique(result)) == 1:
                raise ValueError('Image for {} is blank'.format(word))
                

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

    def _struc_similatity(self, A, B):
        """Computes the structural similarity btw 2 vectors."""

        try:
            return np.array([compare_ssim(v1, v2, win_size=33) for v1, v2 in zip(A, B)])
        except AttributeError as e:
            return 0
        

    def _wasserstein(self, A, B):
        """Computes the  earth mover's distance btw 2 vectors."""
        try:
            return np.array([1 - wasserstein_distance(v1, v2) for v1, v2 in zip(A, B)])
        except TypeError as e:
            return 0
        
        
    def _similarbits(self, A, B):
        """Computes the similar bits btw 2 vectors."""

        try:
            C = A*B
            return np.array([v1.sum() for v1 in C])  # sum number of 1 bits
        except TypeError as e:
            return 0
        
