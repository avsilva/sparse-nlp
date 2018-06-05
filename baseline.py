# coding: utf-8
import sys, os
from sqlalchemy import create_engine
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import utils.evaluation as eval
from sklearn.datasets.base import Bunch
import numpy as np
import scipy
from scipy import spatial
from scipy.sparse import csr_matrix
import pickle


def get_baseline_vector(_word, voc, X):
    try:
        key = voc[_word]

        #if we want to get poor results
        #key = voc[_word] + 1
        #print (X[:, key])
        #print (X[:, key].toarray().flatten()[37])
    except:
        key = voc[_word]
    return X[:, key].toarray().flatten()


def get_dataframe():
    conn_string = 'postgresql://postgres@localhost:5432/sparsenlp'
    engine = create_engine(conn_string)
    #sql = "select id, cleaned_text from snippets where cleaned = 't'  limit 10000"
    sql = "select id, cleaned_text from snippets where cleaned = 't'"
    return pd.read_sql_query(sql, con=engine)


def tfidf_vectorizer(dictionary, _datasetname, train_data, reuse=True):
    
    #reuse = False

    if (os.path.isfile('./serializations/baseline/X_'+_datasetname+'_baseline_tfidf.npz') is True and reuse): 
        loader = np.load('./serializations/baseline/X_'+_datasetname+'_baseline_tfidf.npz')
        X = csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])
        file = open('./serializations/baseline/vocabulary_'+_datasetname+'.pkl', 'rb')
        voc = pickle.load(file)
        file.close()
    else:
        vectorizer = TfidfVectorizer(
                                    lowercase=True,
                                    vocabulary=dictionary,
                                    stop_words='english',
                                    smooth_idf=True,
                                    sublinear_tf=True,
                                    use_idf=True)
        train_data = dataframe.cleaned_text
        print("vectorizing texts ...")
        
        X = vectorizer.fit_transform(train_data)
        voc = vectorizer.vocabulary_
        if reuse:
            with open('./serializations/baseline/vocabulary_'+_datasetname+'.pkl', 'wb') as f:
                pickle.dump(voc, f)
            np.savez('./serializations/baseline/X_'+_datasetname+'_baseline_tfidf.npz', data=X.data, indices=X.indices, indptr=X.indptr, shape=X.shape)
        print("vectorizing texts done")
    return voc, X


def cosine(A, B):
    result = []
    all_zeros = 0
    index = 0
    indexes_to_remove = []
    for v1, v2 in zip(A, B):
        if np.any(v1) and np.any(v2):
            a = 1 - spatial.distance.cosine(v1, v2)
            result.append(a)
        else:
            all_zeros += 1
            indexes_to_remove.append(index)
        index += 1
    
    print('there are {} all zero vectores'.format(all_zeros))
    return np.array(result), indexes_to_remove


def fetch_MEN(_percentage):

    filepath = DIR+'/similarity/EN-MEN-LEM.txt'
    file = open(filepath, 'r', encoding='utf-8')
    score = []
    w1 = []
    w2 = []
    for line in file:
        data = eval.get_words_for_men_dataset(line)
        w1.append(data[0])
        w2.append(data[1])
        score.append(data[2])

    words = w1 + w2
    dictionary = set(words)

    if _percentage is not None:
        w1, w2, score = eval.get_percentage_records(w1, w2, score, _percentage)

    df = pd.DataFrame({0: w1, 1: w2, 2: score})
    bunch = Bunch(X=df.values[:, 0:2].astype("object"), y=df.values[:, 2:].astype(np.float) / 5.0)
     
    return bunch, dictionary


def fetch_WS353(_percentage):
    filepath = DIR+'/similarity/EN-WS353.txt'
    file = open(filepath, 'r', encoding='utf-8')
    score = []
    w1 = []
    w2 = []
    nline = 1
    for line in file:
        if nline != 1:
            data = eval.get_words_for_ws353_dataset(line)
            w1.append(data[0])
            w2.append(data[1])
            score.append(data[2])
        nline += 1

    words = w1 + w2
    dictionary = set(words)
    if _percentage is not None:
        w1, w2, score = eval.get_percentage_records(w1, w2, score, _percentage)

    df = pd.DataFrame({0: w1, 1: w2, 2: score})
    bunch = Bunch(X=df.values[:, 0:2].astype("object"), y=df.values[:, 2:].astype(np.float))
    return bunch, dictionary


def fetch_ENTruk(_percentage):
    filepath = DIR+'/similarity/EN-TRUK.txt'
    file = open(filepath, 'r', encoding='utf-8')
    score = []
    w1 = []
    w2 = []
    for line in file:
        data = eval.get_words_for_truk_dataset(line)
        w1.append(data[0])
        w2.append(data[1])
        score.append(data[2])

    words = w1 + w2
    dictionary = set(words)
    if _percentage is not None:
        w1, w2, score = eval.get_percentage_records(w1, w2, score, _percentage)

    df = pd.DataFrame({0: w1, 1: w2, 2: score})
    bunch = Bunch(X=df.values[:, 0:2].astype("object"), y=df.values[:, 2:].astype(np.float))
    return bunch, dictionary


def fetch_ENRG65(_percentage):
    filepath = DIR+'/similarity/EN-RG-65.txt'
    file = open(filepath, 'r', encoding='utf-8')
    score = []
    w1 = []
    w2 = []
    for line in file:
        data = eval.get_words_for_rg65_dataset(line)
        w1.append(data[0])
        w2.append(data[1])
        score.append(data[2])

    words = w1 + w2
    dictionary = set(words)
    if _percentage is not None:
        w1, w2, score = eval.get_percentage_records(w1, w2, score, _percentage)

    df = pd.DataFrame({0: w1, 1: w2, 2: score})
    bunch = Bunch(X=df.values[:, 0:2].astype("object"), y=df.values[:, 2:].astype(np.float))
        
    return bunch, dictionary

DATASETS = {
        "men-dataset": fetch_MEN,
        "WS353-dataset": fetch_WS353,
        #"SIMLEX999-dataset": fetch_SimLex999,
        "ENTruk-dataset": fetch_ENTruk,
        "EN-RG-65-dataset": fetch_ENRG65
    }

DIR = 'C:/Users/andre.silva/web_data/'

if __name__ == "__main__":
    
    if len(sys.argv) != 3:
        print("wrong number of arguments")
        print("python .\baseline.py <percentage> <dataset name>")
        sys.exit()

    filepath = 'C:/Users/andre.silva/web_data/similarity/EN-MEN-LEM.txt'
    percentage = int(sys.argv[1])
    dataset = sys.argv[2]

    # w1, w2, score, dictionary = get_benchmark_data(filepath, percentage)
    bunch, dictionary = DATASETS[dataset](percentage)

    print('dictionary len is: {}'.format(len(dictionary)))
    print("Sample data from {}: pair \"{}\" and \"{}\" is assigned score {}".format('MEN dataset', bunch.X[0][0], bunch.X[0][1], bunch.y[0]))
    print('number of compared pairs: {}'.format(bunch.X.shape))

    dataframe = get_dataframe()
    print(dataframe.shape)
    train_data = dataframe.cleaned_text
    voc, X = tfidf_vectorizer(dictionary, dataset, train_data)
    print(X.shape)
    print('getting baseline vectors')

    A = np.vstack(get_baseline_vector(word, voc, X) for word in bunch.X[:, 0])
    B = np.vstack(get_baseline_vector(word, voc, X) for word in bunch.X[:, 1])
    print(A.shape, B.shape)

    print('baseline vectors done')

    scores1, indexes_to_remove = cosine(A, B)

    print('There are {} indexes to remove'.format(len(indexes_to_remove)))

    # y = [x for i, x in iterate(bunch.y)]
    y = [x for i, x in enumerate(bunch.y) if i not in indexes_to_remove]
    print('final cosine similarity score is: {}'.format(scipy.stats.spearmanr(scores1, y).correlation))

    # scores1 = ef.cosine(A, B)
    # scores2 = ef.euclidean(A, B)
    # print('final euclidean distance score is: {}'.format(scipy.stats.spearmanr(scores2, bunch.y).correlation))

# python .\baseline.py 25 men-dataset
# python .\baseline.py 100 EN-RG-65-dataset
# python .\baseline.py 100 WS353-dataset
# python .\baseline.py 100 ENTruk-dataset

