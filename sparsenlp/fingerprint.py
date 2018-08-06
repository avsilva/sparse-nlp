import os
import sys
import numpy as np
import pickle
from PIL import Image
import operator
from functools import reduce
import pandas as pd
from sklearn.datasets.base import Bunch
from scipy.spatial import distance
import scipy
from minisom import MiniSom
sys.path.append(os.path.abspath('./utils'))
import decorators as decorate


class FingerPrint():
    
    def __init__(self, opts):
        """Initializes a FingerPrint instance.

        """
        self.opts = opts
        
    def create_fingerprints(self, snippets_by_word, words):
        
        H = int(self.opts['size'])
        W = int(self.opts['size'])
        N = int(self.opts['n_components'])
        if isinstance(words, str):
            words = words.split(',')

        if self.opts['algorithm'] in ['MINISOMBATCH', 'MINISOMRANDOM']:
            
            with open('./serializations/codebook_{}.npy'.format(self.opts['id']), 'rb') as handle:
                codebook = pickle.load(handle)
            SOM = MiniSom(H, W, N, sigma=1.0, random_seed=1)
            SOM.load_weights(codebook)

            with open('./serializations/X_{}.npz'.format(self.opts['id']), 'rb') as handle:
                X = pickle.load(handle)
            
            for word in words:
                word_counts = snippets_by_word[word]
                #print(word_counts)
                a = np.zeros((H, W), dtype=np.int)

                for info in word_counts[1:]:
                    # print (info)
                    idx = info['idx']
                    bmu = SOM.winner(X[idx])
                    a[bmu[0], bmu[1]] += info['counts']

                a = self._sparsify_fingerprint(a)
                self._create_fp_image(a, word, 'fp_{}'.format(self.opts['id']))

    def _sparsify_fingerprint(self, a):
        
        #hist = np.histogram(a, bins=10, range=None, normed=False, weights=None, density=None)
        hist = np.histogram(a, bins='scott', range=None, normed=False, weights=None, density=None)
        #print (hist[0])
        #print (hist[1])

        sparsify_percentage = 0.02
        nvalues = a.shape[0] * a.shape[1]
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
    # TODO: compare par of words
    @decorate.update_result_log
    def evaluate(self, evaluation_set, measure):
        #if isinstance(words, str):
        #    words = words.split(',')

        # Define distance measures
        distance_measures = {
            "cosine": self._cosine, "euclidean": self._euclidean, "similarbits": self._similarbits, "ssim": self._struc_sim
        }

        w1 = evaluation_set[0]
        w2 = evaluation_set[1]
        score = evaluation_set[2]
        
        df = pd.DataFrame({0: w1, 1: w2, 2: score})
        bunch = Bunch(X=df.values[:, 0:2].astype("object"), y=df.values[:, 2:].astype(np.float))

        if measure == 'ssim':
            mode = '2darray'
            A = [self._get_fingerprint_from_image(word, mode) for word in bunch.X[:, 0]]
            B = [self._get_fingerprint_from_image(word, mode) for word in bunch.X[:, 1]]
        else:
            mode = 'flatten'
            A = np.vstack(self._get_fingerprint_from_image(word, mode) for word in bunch.X[:, 0])
            B = np.vstack(self._get_fingerprint_from_image(word, mode) for word in bunch.X[:, 1])

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
            np.place(pix, pix>1, [1])
            pix = pix.flatten()
            result = pix

        return result

    def _cosine(self, A, B):
        """
        computes the distance, and not the similarity. 
        must subtract the value from 1 to get the similarity.
        """
        return np.array([1 - distance.cosine(v1, v2) for v1, v2 in zip(A, B)])

    def _euclidean(self, A, B):
        return np.array([1 / distance.euclidean(v1, v2) for v1, v2 in zip(A, B)])

    def _struc_sim(self, A, B):
        return np.array([ssim(v1, v2, win_size=33) for v1, v2 in zip(A, B)])
        
    def _similarbits(self, A, B):
        C = A*B
        return np.array([v1.sum() for v1 in C])  # sum number of 1 bits



                
                
