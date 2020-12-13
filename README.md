# SparseNLP

## Introduction

- Word Embeddings are an important component of many language models capable of producing
 state of the art results in several NLP tasks;
- SparseNLP proposes an alternative approach for deriving word embeddings. In contrast with the tradicional dense vector representations, it creates sparse distributed representations (SDR) for each word; the representation shares the main idea of a word embedding, which was formulated by Firth in 1957: "a word is characterized by the company it keeps";
- Validation of the whole methodology is done by using these word embeddings as language models in several Natural Language Processing tasks;


## Installation

sudo apt-get update

sudo apt-get install zip

sudo apt remove python3
sudo apt remove python
wget https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh
bash Anaconda3-5.1.0-Linux-x86_64.sh
rm Anaconda3-5.1.0-Linux-x86_64.sh

git clone https://github.com/avsilva/sparse-nlp.git

mkdir sparse-nlp/serializations
mkdir sparse-nlp/serializations/sentences
mkdir sparse-nlp/images
mkdir sparse-nlp/logs
mkdir sparse-nlp/datasets
mkdir sparse-nlp/datasets/analogy
mkdir sparse-nlp/datasets/similarity
mkdir sparse-nlp/embeddings
mkdir sparse-nlp/embeddings/glove.6B

mkdir wikiextractor
mkdir wikiextractor/jsonfiles
mkdir wikiextractor/jsonfiles/articles3

cd sparse-nlp

pip install -r requirements.txt 
conda install --yes --file requirements2.txt

python -m spacy download en

## How to use it

In order to use SparseNLP you need your data stored in a database table with 2 columns: 
- id (int): primary key
-  cleaned_text (str): text tokens for each sentence

## Tests

- nosetests --cover-package=.\sparsenlp --with-coverage --nologcapture -x
- python -m unittest -v

  (run just one class test)
- python -m unittest -q tests.test_datacleaner.TestDataClean
- py.test -q -s tests/test_datacleaner.py::TestDataClean

 (run just one functional test)

 - python -m unittest -q tests.test_datacleaner.TestDataClean.test_ingestfiles_json_to_dict


https://realpython.com/fast-flexible-pandas/
https://towardsdatascience.com/stop-using-pandas-and-start-using-spark-with-scala-f7364077c2e0
https://www.kdnuggets.com/2019/11/speed-up-pandas-4x.html

## TODO
evaluate using word-embeddings-benchmarks
train word2vec on wikidumps


## Project Planning

0. Extracting data
1. Training Corpora Definition 
2. Corpora pre-processing
3. Sentence tokenization
4. Sentence vetorization
5. Word to sentence database
6. Cluster sentences
7. Word fingerprint
8. Text fingerprint
9. Evaluation
10. Trainning Word2Vec word embeedings

## 1. Training Corpora Definition

Wikipedia dumps from [wikimedia 2018-01-01](https://dumps.wikimedia.org/enwiki/20180101/) 

### 1.1 Extracting plain text from Wikipedia dumps

[github - attardi/wikiextractor](https://github.com/attardi/wikiextractor)

Document files contains a series of Wikipedia articles, represented each by an XML doc element:
```markdown
<doc>...</doc>
<doc>...</doc>
...
<doc>...</doc>
```
The element doc has the following attributes:

- id, which identifies the document by means of a unique serial number
- url, which provides the URL of the original Wikipedia page.
The content of a doc element consists of pure text, one paragraph per line.

Example:
```markdown
<doc id="2" url="http://it.wikipedia.org/wiki/Harmonium">
Harmonium.
L'harmonium è uno strumento musicale azionato con una tastiera, detta manuale.
Sono stati costruiti anche alcuni harmonium con due manuali.
...
</doc>
```


## 9. Evaluation

Evaluation code repository: [github - kudkudak/word-embeddings-benchmarks](https://github.com/kudkudak/word-embeddings-benchmarks.git)
Evaluation methods: [arxiv.org/abs/1702.02170](https://arxiv.org/abs/1702.02170)

other alternative methods: [github - mfaruqui/eval-word-vectors](https://github.com/mfaruqui/eval-word-vectors)





