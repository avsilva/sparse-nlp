# coding: utf-8
import sys, os
from sqlalchemy import create_engine
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import utils.evaluation as eval
import utils.database as db
from sklearn.datasets.base import Bunch
import numpy as np
import scipy
from scipy import spatial
from scipy.sparse import csr_matrix
import pickle


def get_baseline_vector(_word, voc, X):
    try:
        key = voc[_word]
    except:
        #key = voc[_word]
        print (_word)
        return X[:, 1].toarray().flatten()
    return X[:, key].toarray().flatten()


def get_dataframe():
    conn_string = 'postgresql://postgres@localhost:5432/sparsenlp'
    engine = create_engine(conn_string)
    #sql = "select id, cleaned_text from snippets where cleaned = 't'  limit 10000"
    sql = "select id, cleaned_text from snippets where cleaned = 't'"
    return pd.read_sql_query(sql, con=engine)


def tfidf_vectorizer(dictionary, _datasetname, reuse=True):
    
    #reuse = False

    if (os.path.isfile('./serializations/baseline/X_'+_datasetname+'_baseline_tfidf.npz') is True and reuse): 
        loader = np.load('./serializations/baseline/X_'+_datasetname+'_baseline_tfidf.npz')
        X = csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])
        file = open('./serializations/baseline/vocabulary_'+_datasetname+'.pkl', 'rb')
        voc = pickle.load(file)
        file.close()
    else:
        dataframe = db.get_cleaned_data(None, None)
        print(dataframe.shape)
        train_data = dataframe.cleaned_text
        vectorizer = TfidfVectorizer(
                                    lowercase=True,
                                    #vocabulary=dictionary,
                                    #max_df=0.5, 
                                    #min_df=0.001, 
                                    stop_words='english',
                                    smooth_idf=True,
                                    sublinear_tf=True,
                                    use_idf=True)
        
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


def wiki_tfidf(bunch, dictionary, datasetname):
    
    voc, X = tfidf_vectorizer(dictionary, datasetname)
    print(X.shape)
    print('getting baseline vectors')

    A = np.vstack(get_baseline_vector(word, voc, X) for word in bunch.X[:, 0])
    B = np.vstack(get_baseline_vector(word, voc, X) for word in bunch.X[:, 1])
    return A, B


def gimme_glove():
    with open(DIR+'/embeddings/glove.6B/glove.6B.50d.txt', encoding='utf-8') as glove_raw:
        for line in glove_raw.readlines():
            splitted = line.split(' ')
            yield splitted[0], np.array(splitted[1:], dtype=np.float)


def glove(bunch, dictionary, datasetname):
    glove = {w: x for w, x in gimme_glove()}

    print('getting glove vectors')
    A = np.vstack(np.array(glove[word]) for word in bunch.X[:, 0])
    B = np.vstack(np.array(glove[word]) for word in bunch.X[:, 1])
    return A, B


DATASETS = {
        "men-dataset": fetch_MEN,
        "WS353-dataset": fetch_WS353,
        "ENTruk-dataset": fetch_ENTruk,
        "EN-RG-65-dataset": fetch_ENRG65
    }

VECTORS = {
        "wiki_tfidf": wiki_tfidf,
        "glove": glove,
    }

DIR = 'C:/Users/andre.silva/web_data/'

if __name__ == "__main__":
    
    if len(sys.argv) != 4:
        print("wrong number of arguments")
        print("python .\baseline.py <percentage> <dataset name> <vectors>")
        sys.exit()

    percentage = int(sys.argv[1])
    dataset = sys.argv[2]
    vectors = sys.argv[3]

    bunch, dictionary = DATASETS[dataset](percentage)

    print('dictionary len is: {}'.format(len(dictionary)))
    print("Sample data from {}: pair \"{}\" and \"{}\" is assigned score {}".format('RG 65 dataset', bunch.X[0][0], bunch.X[0][1], bunch.y[0]))
    print('number of compared pairs: {}'.format(bunch.X.shape))

    print (vectors)
    A, B = VECTORS[vectors](bunch, dictionary, dataset)
    
    #print(A.shape, B.shape)
    print('baseline vectors done')

    scores1, indexes_to_remove = cosine(A, B)

    print('There are {} indexes to remove'.format(len(indexes_to_remove)))

    # y = [x for i, x in iterate(bunch.y)]
    y = [x for i, x in enumerate(bunch.y) if i not in indexes_to_remove]
    print('final cosine similarity score is: {}'.format(scipy.stats.spearmanr(scores1, y).correlation))

    # scores1 = ef.cosine(A, B)
    # scores2 = ef.euclidean(A, B)




# python .\baseline.py 25 men-dataset
# python .\baseline.py 100 WS353-dataset
# python .\baseline.py 100 ENTruk-dataset


# python .\baseline.py 100 EN-RG-65-dataset
# python .\baseline.py 100 EN-RG-65-dataset wiki_tfidf
# python .\baseline.py 100 EN-RG-65-dataset glove

