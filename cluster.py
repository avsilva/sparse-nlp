# coding: utf-8
import os, json, sys, csv, shutil
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
from time import time
import pickle

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets import fetch_20newsgroups

from sparse_som import *
from minisom import MiniSom
from scipy import sparse


# Do the actual doc representation
def doc_representation(train_data, opts):
    
    if opts['use_hashing']:
        if opts['use_idf']:
            # Perform an IDF normalization on the output of HashingVectorizer
            hasher = HashingVectorizer(n_features=opts['n_features'],
                                    stop_words='english', alternate_sign=False,
                                    norm=None, binary=False)
            vectorizer = make_pipeline(hasher, TfidfTransformer())
        else:
            vectorizer = HashingVectorizer(n_features=opts['n_features'],
                                        stop_words='english',
                                        alternate_sign=False, norm='l2',
                                        binary=False)
    else:
        vectorizer = TfidfVectorizer(max_df=0.5, max_features=opts['n_features'],
                                    min_df=2, stop_words='english',
                                    use_idf=opts['use_idf'])
        
    X = vectorizer.fit_transform(train_data)

    if opts['n_components']:
        print("Performing dimensionality reduction using LSA")
        t0 = time()
        # Vectorizer results are normalized, which makes KMeans behave as
        # spherical k-means for better results. Since LSA/SVD results are
        # not normalized, we have to redo the normalization.
        svd = TruncatedSVD(opts['n_components'])
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)

        X = lsa.fit_transform(X)

        print("done in %fs" % (time() - t0))

        explained_variance = svd.explained_variance_ratio_.sum()
        print("Explained variance of the SVD step: {}%".format(
            int(explained_variance * 100)))

    return (X)


# Do the actual clustering
def clustering(X, true_k, opts):
    if opts['minibatch']:
        km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                            init_size=1000, batch_size=1000, verbose=opts['verbose'])
    else:
        km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1,
                    verbose=opts['verbose'])

    print("Clustering sparse data with %s" % km)
    t0 = time()
    km.fit(X)
    print("done in %0.3fs" % (time() - t0))
    return km


def evaluate(km, labels):
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
    print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
    print("Adjusted Rand-Index: %.3f" % metrics.adjusted_rand_score(labels, km.labels_))
    #print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, km.labels_, sample_size=1000))


def get_dataset():
    engine = create_engine('postgresql://postgres@localhost:5432/sparsenlp')
    sql = "select id, text, label from news"
    dataframe = pd.read_sql_query(sql, con=engine)
    return dataframe

def predict_with_SOM(X, target, Y):
    
    H, W = 25, 25   # Network height and width
    N = X.shape[1]  # Nb. features (vectors dimension)
    # setup SOM classifier (using batch SOM)
    cls = SomClassifier(Som, H, W, N)

    # use SOM calibration
    cls.fit(X, labels=target)

    # make predictions
    y = cls.predict(Y)
    return y


def evaluate_som(X, labels):
    X_train, X_test, y_train, y_test = train_test_split(X, labels, 
                                                        test_size=0.3, 
                                                        random_state=42)
    
    X_train = sparse.csr_matrix(X_train)
    X_test = sparse.csr_matrix(X_test)
    
    predicted = predict_with_SOM(X_train, y_train, X_test)
    return np.mean(predicted == y_test)
    #print(classification_report(y_test, y))
    

def generate_sparse_som(X, algo):
    
    X = sparse.csr_matrix(X)
    # setup SOM dimensions
    H, W = 25, 25   # Network height and width
    N = X.shape[1]  # Nb. features (vectors dimension)
    print('number of features in SOM: {}'.format(N))
    som_type = {'SDSOM': Som, 'BSOM': BSom}
    # setup SOM network
    som = som_type[algo](H, W, N, topology.RECT, verbose=True)
    #som = Som(H, W, N, topology.RECT, verbose=True) # , verbose=True
    # reinit the codebook (not needed)
    som.codebook = np.random.rand(H, W, N).astype(som.codebook.dtype, copy=False)

    # train the SOM
    t1=time()
    tmax = 10*X.shape[0]
    som.train(X, tmax=tmax)
    t2=time()
    print("\nTime taken by training standard sparse som\n----------\n{} s".format((t2-t1)))
    return som


def generate_minisom(X, algo):
    
    map_dim = 25
    N = X.shape[1]  # Nb. features (vectors dimension)
    print('number of features in SOM: {}'.format(N))
    som = MiniSom(map_dim, map_dim, N, sigma=1.0, random_seed=1)
    #som.random_weights_init(X)
    t1=time()
    #som.train_batch(X, 10*X.shape[0])
    if algo == 'BATCH':
        som.train_batch(X, 500)
    elif algo == 'RANDOM':
        som.train_random(X, 500)
    t2=time()
    print("\nTime taken by training {} minisom\n----------\n{} s".format(algo, (t2-t1)))


if __name__ == "__main__":
    opts = {}
    opts['n_components'] = 700
    #opts['n_components'] = False
    opts['use_hashing'] = False
    opts['use_idf'] = True
    opts['n_features'] = 10000
    opts['minibatch'] = False
    opts['verbose'] = False

    dataframe = get_dataset()
    labels = dataframe.label
    true_k = np.unique(labels).shape[0]


    filepath = './tests/X_test.pkl'
    if not os.path.isfile(filepath):
        X = doc_representation(dataframe.text, opts)
        with open('./tests/X_test.pkl', 'wb') as f:
            pickle.dump(X, f)
    else:
        with open('./tests/X_test.pkl', 'rb') as handle:
            X = pickle.load(handle)
        
        
    # K-means
    #y_train = dataframe.label
    #km = clustering(X, true_k, opts)
    #evaluate(km, y_train)

    # evaluate SOM
    #result = evaluate_som(X, dataframe.label)
    #print (result)


    #sparse_som = generate_sparse_som(X, 'SDSOM')    
    ##108.84931111335754 s

    minisom = generate_minisom(X, 'RANDOM')

    

# testar somoclu
# testar import wikipedia
# testar t-sne
# testar visualizacao da matriz SOM
# SVD explained variance !!!











## tentar calcular accuracy_score: sem sucesso
#X = doc_representation(dataframe.text, opts)
#X_train, X_test, y_train, y_test = train_test_split(X, dataframe.label, 
#                                                    test_size=0.3, 
#                                                    random_state=42)
# y_labels_train = km.labels_
#y_pred = km.predict(X_test)
# print('Accuracy: {}'.format(accuracy_score(y_test, y_pred)))
# print(classification_report(y_test, y_pred))
    
    